"""MCD: Maximum Classifier Discrepancy (Saito et al., CVPR 2018)."""
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .training import evaluate, build_scheduler


def _discrepancy(p1, p2):
    return (p1 - p2).abs().mean()


def _tgt_imgs(batch, device):
    if isinstance(batch, (list, tuple)):
        return batch[0].to(device)
    return batch.to(device)


def run_mcd(model, cfg, device,
            src_loader, src_eval_loader, tgt_loader, eval_loader,
            wandb_run=None):
    """
    3-step MCD per iteration:
      A) Train F + C1 + C2 with source CE
      B) Fix F, maximize discrepancy on target (still supervised on source)
      C) Fix C1/C2, minimize discrepancy by training F  (repeated k times)
    Returns best target accuracy seen during training.
    """
    assert model.classifier2 is not None, "MCD requires two_classifier=True"

    src_epochs = cfg.get('src_epochs', 80)
    k          = cfg.get('mcd_k', 4)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get('label_smoothing', 0.0))

    lr  = cfg['lr']
    mom = cfg.get('momentum', 0.9)
    wd  = cfg.get('weight_decay', 5e-4)

    if cfg.get('optimizer', 'sgd') == 'adamw':
        wd       = cfg.get('weight_decay', 1e-2)
        feat_opt = torch.optim.AdamW(model.feat_parameters(), lr=lr, weight_decay=wd)
        cls_opt  = torch.optim.AdamW(model.cls_parameters(),  lr=lr, weight_decay=wd)
    else:
        feat_opt = torch.optim.SGD(model.feat_parameters(), lr=lr, momentum=mom, weight_decay=wd)
        cls_opt  = torch.optim.SGD(model.cls_parameters(),  lr=lr, momentum=mom, weight_decay=wd)

    feat_sched = build_scheduler(feat_opt, cfg, src_epochs)
    cls_sched  = build_scheduler(cls_opt,  cfg, src_epochs)

    best_tgt = 0.0

    for epoch in range(src_epochs):
        model.train()
        tgt_iter = itertools.cycle(tgt_loader)
        disc_sum = n = 0

        for src_imgs, src_lbls in tqdm(src_loader, desc=f"[MCD] ep{epoch+1}", leave=False):
            src_imgs = src_imgs.to(device)
            src_lbls = src_lbls.to(device)
            tgt_imgs = _tgt_imgs(next(tgt_iter), device)

            # ── Step A: Train F + C1 + C2 on source ──────────────────────────
            feat_opt.zero_grad(); cls_opt.zero_grad()
            f_src  = model.encode(src_imgs)
            loss_a = (criterion(model.classifier(f_src),  src_lbls) +
                      criterion(model.classifier2(f_src), src_lbls))
            loss_a.backward()
            feat_opt.step(); cls_opt.step()

            # ── Step B: Fix F, maximize discrepancy on target ─────────────────
            cls_opt.zero_grad()
            with torch.no_grad():
                f_src_d = model.encode(src_imgs)
                f_tgt_d = model.encode(tgt_imgs)
            p1_t = F.softmax(model.classifier(f_tgt_d),  dim=1)
            p2_t = F.softmax(model.classifier2(f_tgt_d), dim=1)
            disc_b  = _discrepancy(p1_t, p2_t)
            src_ce  = (criterion(model.classifier(f_src_d),  src_lbls) +
                       criterion(model.classifier2(f_src_d), src_lbls))
            (src_ce - disc_b).backward()
            cls_opt.step()

            # ── Step C: Fix C1/C2, minimize discrepancy (train F only) ────────
            for _ in range(k):
                feat_opt.zero_grad()
                f_tgt = model.encode(tgt_imgs)
                p1 = F.softmax(model.classifier(f_tgt),  dim=1)
                p2 = F.softmax(model.classifier2(f_tgt), dim=1)
                _discrepancy(p1, p2).backward()
                feat_opt.step()

            disc_sum += disc_b.item()
            n += 1

        if feat_sched: feat_sched.step()
        if cls_sched:  cls_sched.step()

        tgt_acc = evaluate(model, eval_loader, device)
        src_acc = evaluate(model, src_eval_loader, device)
        best_tgt = max(best_tgt, tgt_acc)

        print(f"  MCD ep{epoch+1}/{src_epochs}  disc={disc_sum/n:.4f}"
              f"  src={src_acc:.2f}%  tgt={tgt_acc:.2f}%")

        if wandb_run:
            wandb_run.log({
                'mcd/epoch':       epoch + 1,
                'mcd/discrepancy': disc_sum / n,
                'mcd/src_acc':     src_acc,
                'mcd/tgt_acc':     tgt_acc,
                'mcd/best_tgt':    best_tgt,
            }, step=epoch + 1)

    return best_tgt
