"""DANN: Domain-Adversarial Neural Network (Ganin et al., ICML 2016)."""
import itertools
import math
import torch
import torch.nn as nn
from tqdm import tqdm

from .training import evaluate, build_optimizer, build_scheduler


def _tgt_imgs(batch, device):
    """Handle both single-tensor and two-view tuple loaders."""
    if isinstance(batch, (list, tuple)):
        return batch[0].to(device)
    return batch.to(device)


def run_dann(model, discriminator, cfg, device,
             src_loader, src_eval_loader, tgt_loader, eval_loader,
             wandb_run=None):
    """
    Joint source CE + domain adversarial training.
    GRL automatically reverses feature-extractor gradients.
    Returns best target accuracy seen during training.
    """
    src_epochs  = cfg.get('src_epochs', 80)
    lambda_dann = cfg.get('lambda_dann', 1.0)
    n_total     = src_epochs * len(src_loader)

    criterion   = nn.CrossEntropyLoss(label_smoothing=cfg.get('label_smoothing', 0.0))
    domain_crit = nn.BCEWithLogitsLoss()

    # One optimizer covers both model and discriminator;
    # GRL handles the adversarial reversal automatically.
    all_params = list(model.parameters()) + list(discriminator.parameters())
    optim_cfg  = dict(cfg)
    if cfg.get('optimizer', 'sgd') == 'adamw':
        optimizer = torch.optim.AdamW(
            all_params, lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 1e-2))
    else:
        optimizer = torch.optim.SGD(
            all_params, lr=cfg['lr'],
            momentum=cfg.get('momentum', 0.9),
            weight_decay=cfg.get('weight_decay', 5e-4))
    scheduler = build_scheduler(optimizer, cfg, src_epochs)

    best_tgt = 0.0
    step     = 0

    for epoch in range(src_epochs):
        model.train(); discriminator.train()
        tgt_iter = itertools.cycle(tgt_loader)
        ce_sum = dom_sum = n = 0

        for src_imgs, src_lbls in tqdm(src_loader, desc=f"[DANN] ep{epoch+1}", leave=False):
            src_imgs  = src_imgs.to(device)
            src_lbls  = src_lbls.to(device)
            tgt_imgs  = _tgt_imgs(next(tgt_iter), device)

            # Scheduled alpha: ramps 0→1 over training (Ganin et al.)
            p     = step / n_total
            alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
            step += 1

            optimizer.zero_grad()

            src_feat = model.encode(src_imgs)
            tgt_feat = model.encode(tgt_imgs)

            ce_loss = criterion(model.classifier(src_feat), src_lbls)

            src_dom = discriminator(src_feat, alpha)
            tgt_dom = discriminator(tgt_feat, alpha)
            dom_loss = 0.5 * (
                domain_crit(src_dom, torch.ones_like(src_dom)) +
                domain_crit(tgt_dom, torch.zeros_like(tgt_dom))
            )

            (ce_loss + lambda_dann * dom_loss).backward()
            optimizer.step()

            ce_sum  += ce_loss.item()
            dom_sum += dom_loss.item()
            n += 1

        if scheduler:
            scheduler.step()

        tgt_acc = evaluate(model, eval_loader, device)
        src_acc = evaluate(model, src_eval_loader, device)
        best_tgt = max(best_tgt, tgt_acc)

        print(f"  DANN ep{epoch+1}/{src_epochs}  CE={ce_sum/n:.4f}"
              f"  DOM={dom_sum/n:.4f}  src={src_acc:.2f}%  tgt={tgt_acc:.2f}%"
              f"  alpha={alpha:.3f}")

        if wandb_run:
            wandb_run.log({
                'dann/epoch':       epoch + 1,
                'dann/ce_loss':     ce_sum / n,
                'dann/domain_loss': dom_sum / n,
                'dann/src_acc':     src_acc,
                'dann/tgt_acc':     tgt_acc,
                'dann/best_tgt':    best_tgt,
                'dann/alpha':       alpha,
            }, step=epoch + 1)

    return best_tgt
