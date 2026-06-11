"""SHOT adaptation (Liang et al., ICML 2020) with wandb logging."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .data import UnlabeledDataset, PseudoTargetDataset, make_loader
from .training import evaluate, build_optimizer, build_scheduler


def im_loss(logits):
    probs  = F.softmax(logits, dim=1)
    L_ent  = -(probs * (probs + 1e-8).log()).sum(1).mean()
    mean_p = probs.mean(0)
    L_div  = (mean_p * (mean_p + 1e-8).log()).sum()
    return L_ent + L_div


@torch.no_grad()
def extract_features_and_probs(model, loader, device):
    model.eval()
    feats, probs = [], []
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(device)
        f = model.encode(imgs)
        p = F.softmax(model.classifier(f), dim=1)
        feats.append(f.cpu()); probs.append(p.cpu())
    return torch.cat(feats), torch.cat(probs)


def compute_prototype_pseudo_labels(features, probs, num_classes=200):
    feats_n    = F.normalize(features, dim=1)
    D          = features.shape[1]
    prototypes = torch.zeros(num_classes, D)
    for c in range(num_classes):
        w             = probs[:, c]
        prototypes[c] = (w.unsqueeze(1) * feats_n).sum(0) / (w.sum() + 1e-8)
    prototypes = F.normalize(prototypes, dim=1)
    return (feats_n @ prototypes.T).argmax(dim=1)


def shot_epoch(model, loader, optimizer, device, lambda_im, lambda_pseudo):
    model.train()
    im_sum = ce_sum = n = 0
    for imgs, pseudo_lbls, _ in tqdm(loader, desc="[SHOT]", leave=False):
        imgs, pseudo_lbls = imgs.to(device), pseudo_lbls.to(device)
        optimizer.zero_grad()
        logits  = model(imgs)
        loss    = lambda_im * im_loss(logits) + lambda_pseudo * F.cross_entropy(logits, pseudo_lbls)
        loss.backward()
        optimizer.step()
        im_sum += (lambda_im * im_loss(logits.detach())).item()
        ce_sum += F.cross_entropy(logits.detach(), pseudo_lbls).item()
        n += 1
    return im_sum / n, ce_sum / n


def run_shot_adaptation(model, tgt_root, tgt_tf, eval_loader, src_eval_loader,
                         cfg, device, wandb_run=None):
    """
    Returns best_tgt_acc achieved during all SHOT rounds.
    Logs per-epoch metrics to wandb if wandb_run is provided.
    """
    num_classes  = 200
    shot_rounds  = cfg.get('shot_rounds', 3)
    shot_epochs  = cfg.get('shot_epochs', 15)
    lambda_im    = cfg.get('lambda_im', 1.0)
    lambda_pseudo = cfg.get('lambda_pseudo', 1.0)
    global_step  = cfg.get('_global_step', 0)

    for p in model.classifier.parameters():
        p.requires_grad_(False)

    feat_ds     = UnlabeledDataset(ImageFolder(root=tgt_root, transform=tgt_tf))
    feat_loader = make_loader(feat_ds, cfg['batch_size'], shuffle=False,
                               num_workers=cfg.get('num_workers', 4))

    shot_cfg = dict(cfg)
    shot_cfg['lr'] = cfg.get('shot_lr', 1e-4)
    optimizer = build_optimizer(model, shot_cfg)

    best_tgt = 0.0

    for rnd in range(1, shot_rounds + 1):
        features, probs = extract_features_and_probs(model, feat_loader, device)
        pseudo_labels   = compute_prototype_pseudo_labels(features, probs, num_classes)

        true_labels = torch.tensor([feat_ds.dataset.targets[i] for i in range(len(feat_ds))])
        pl_acc = (pseudo_labels == true_labels).float().mean().item() * 100

        pseudo_ds = PseudoTargetDataset(tgt_root, tgt_tf, pseudo_labels)
        loader    = make_loader(pseudo_ds, cfg['batch_size'], shuffle=True,
                                 num_workers=cfg.get('num_workers', 4))
        sched = build_scheduler(optimizer, cfg, shot_epochs)

        for epoch in range(shot_epochs):
            loss_im, loss_ce = shot_epoch(model, loader, optimizer, device,
                                           lambda_im, lambda_pseudo)
            tgt_acc = evaluate(model, eval_loader, device)
            src_acc = evaluate(model, src_eval_loader, device)
            best_tgt = max(best_tgt, tgt_acc)
            global_step += 1

            print(f"  SHOT R{rnd} ep{epoch+1}/{shot_epochs}"
                  f"  pl_acc={pl_acc:.1f}%  tgt={tgt_acc:.2f}%  src={src_acc:.2f}%")

            if wandb_run:
                wandb_run.log({
                    'shot/round': rnd,
                    'shot/epoch': epoch + 1,
                    'shot/im_loss': loss_im,
                    'shot/ce_loss': loss_ce,
                    'shot/pseudo_acc': pl_acc,
                    'shot/tgt_acc': tgt_acc,
                    'shot/src_acc': src_acc,
                    'shot/best_tgt': best_tgt,
                }, step=global_step)

            if sched:
                sched.step()

    cfg['_global_step'] = global_step
    return best_tgt
