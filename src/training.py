"""Joint CE + InfoNCE training with wandb logging."""
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .data import (
    UnlabeledDataset, PseudoLabeledDataset, make_loader, get_transform,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        out = model(imgs.to(device))
        correct += (out.argmax(1).cpu() == labels).sum().item()
        total   += labels.size(0)
    return 100 * correct / total


@torch.no_grad()
def generate_pseudo_labels(model, root, transform, device, threshold, batch_size, num_workers,
                           select_mode='threshold', top_k=15, num_classes=200):
    model.eval()
    ds     = ImageFolder(root=root, transform=transform)
    loader = make_loader(ds, batch_size, shuffle=False, num_workers=num_workers)

    all_probs = []
    for imgs, _ in loader:
        feat = model.encode(imgs.to(device))
        p1   = F.softmax(model.classifier(feat), dim=1)
        # Ensemble two classifiers if available (better pseudo quality)
        if getattr(model, 'classifier2', None) is not None:
            p2 = F.softmax(model.classifier2(feat), dim=1)
            all_probs.append(((p1 + p2) / 2).cpu())
        else:
            all_probs.append(p1.cpu())
    all_probs = torch.cat(all_probs)          # (N, C)
    max_prob, all_preds = all_probs.max(1)

    if select_mode == 'balanced':
        # per-class top-k by confidence → equal class coverage
        indices, labels = [], []
        for c in range(num_classes):
            cls_idx = (all_preds == c).nonzero(as_tuple=True)[0]
            if len(cls_idx) == 0:
                continue
            k = min(top_k, len(cls_idx))
            top_local = max_prob[cls_idx].topk(k).indices
            for j in top_local:
                indices.append(cls_idx[j].item())
                labels.append(c)
    else:
        mask    = (max_prob >= threshold).nonzero(as_tuple=True)[0]
        indices = mask.tolist()
        labels  = all_preds[mask].tolist()

    return indices, labels


# ── Source epoch ──────────────────────────────────────────────────────────────

def train_source_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.0):
    model.train()
    loss_sum = acc_sum = n = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        if mixup_alpha > 0:
            mixed, y_a, y_b, lam = mixup_data(imgs, lbls, mixup_alpha)
            out  = model(mixed)
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            with torch.no_grad():
                acc_sum += model(imgs).argmax(1).eq(lbls).float().mean().item()
        else:
            out  = model(imgs)
            loss = criterion(out, lbls)
            acc_sum += out.detach().argmax(1).eq(lbls).float().mean().item()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item(); n += 1
    return loss_sum / n, acc_sum / n


# ── Joint epoch (CE + InfoNCE + optional EM) ──────────────────────────────────

def train_joint_epoch(model, src_loader, tgt_loader, optimizer, criterion,
                      device, lambda_nce, temperature, noise_std,
                      lambda_em=0.0, mixup_alpha=0.0):
    from info_nce import info_nce
    feat_fn  = model.project if hasattr(model, 'project') else model
    model.train()
    tgt_iter = itertools.cycle(tgt_loader)
    ce_sum = nce_sum = em_sum = acc_sum = n = 0

    for src_imgs, src_lbls in tqdm(src_loader, desc="[Joint]", leave=False):
        src_imgs, src_lbls = src_imgs.to(device), src_lbls.to(device)
        tgt_batch = next(tgt_iter)
        optimizer.zero_grad()

        # Two-view NCE (TwoViewDataset) or fallback to noise augmentation
        if isinstance(tgt_batch, (list, tuple)):
            tgt_v1, tgt_v2 = tgt_batch[0].to(device), tgt_batch[1].to(device)
        else:
            tgt_v1 = tgt_batch.to(device)
            tgt_v2 = tgt_v1 + torch.randn_like(tgt_v1) * noise_std

        if mixup_alpha > 0:
            mixed, y_a, y_b, lam = mixup_data(src_imgs, src_lbls, mixup_alpha)
            feat_m  = model.encode(mixed)
            src_out = model.classifier(feat_m)
            ce_loss = lam * criterion(src_out, y_a) + (1 - lam) * criterion(src_out, y_b)
            if getattr(model, 'classifier2', None) is not None:
                src_out2 = model.classifier2(feat_m)
                ce_loss  = ce_loss + lam * criterion(src_out2, y_a) + (1 - lam) * criterion(src_out2, y_b)
            with torch.no_grad():
                acc_sum += model(src_imgs).argmax(1).eq(src_lbls).float().mean().item()
        else:
            feat    = model.encode(src_imgs)
            src_out = model.classifier(feat)
            ce_loss = criterion(src_out, src_lbls)
            if getattr(model, 'classifier2', None) is not None:
                ce_loss = ce_loss + criterion(model.classifier2(feat), src_lbls)
            acc_sum += src_out.detach().argmax(1).eq(src_lbls).float().mean().item()

        nce_loss = info_nce(feat_fn(tgt_v1), feat_fn(tgt_v2), temperature=temperature)

        em_loss = torch.tensor(0.0, device=device)
        if lambda_em > 0:
            p = F.softmax(model(tgt_v1), dim=1)
            em_loss = -(p * p.log()).sum(1).mean()

        (ce_loss + lambda_nce * nce_loss + lambda_em * em_loss).backward()
        optimizer.step()
        ce_sum += ce_loss.item(); nce_sum += nce_loss.item(); em_sum += em_loss.item(); n += 1

    return ce_sum / n, nce_sum / n, em_sum / n, acc_sum / n


# ── Optimizer / scheduler builders ────────────────────────────────────────────

def build_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.get('optimizer', 'sgd') == 'adamw':
        return torch.optim.AdamW(params, lr=cfg['lr'],
                                  weight_decay=cfg.get('weight_decay', 1e-2))
    return torch.optim.SGD(params, lr=cfg['lr'],
                            momentum=cfg.get('momentum', 0.9),
                            weight_decay=cfg.get('weight_decay', 5e-4))


def build_scheduler(optimizer, cfg, n_epochs):
    sched = cfg.get('scheduler', 'cosine')
    if sched == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    if sched == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.get('step_size', 10), gamma=cfg.get('gamma', 0.1))
    return None
