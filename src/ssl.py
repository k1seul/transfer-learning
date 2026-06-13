"""SimCLR-style SSL pre-training on unlabeled target domain (CUB)."""
import torch
from info_nce import info_nce
from tqdm import tqdm

from .data import TwoViewDataset, get_contrastive_transform, make_loader
from .training import build_optimizer, build_scheduler


def _ssl_epoch(model, loader, optimizer, device, temperature):
    model.train()
    loss_sum = n = 0
    for v1, v2 in tqdm(loader, desc="[SSL]", leave=False):
        v1, v2 = v1.to(device), v2.to(device)
        optimizer.zero_grad()
        loss = info_nce(model.project(v1), model.project(v2), temperature=temperature)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        n += 1
    return loss_sum / n


def run_ssl_pretrain(model, tgt_root, cfg, device, wandb_run=None):
    """SimCLR pre-training on unlabeled CUB images.

    Trains backbone + projector only; classifier weights are frozen and restored
    after pre-training so source training starts from a fresh head.
    """
    ssl_epochs      = cfg.get('ssl_epochs', 100)
    ssl_temperature = cfg.get('ssl_temperature', 0.1)
    ssl_batch_size  = cfg.get('ssl_batch_size', 128)
    num_workers     = cfg.get('num_workers', 4)

    # freeze classifier so optimizer doesn't touch it
    for p in model.classifier.parameters():
        p.requires_grad_(False)
    if getattr(model, 'classifier2', None) is not None:
        for p in model.classifier2.parameters():
            p.requires_grad_(False)

    ssl_cfg = {**cfg,
               'lr':           cfg.get('ssl_lr', 1e-3),
               'optimizer':    cfg.get('ssl_optimizer', 'adamw'),
               'weight_decay': cfg.get('ssl_weight_decay', 1e-4),
               'scheduler':    'cosine'}

    tf        = get_contrastive_transform('cub')
    ds        = TwoViewDataset(tgt_root, tf)
    loader    = make_loader(ds, ssl_batch_size, shuffle=True, num_workers=num_workers)
    optimizer = build_optimizer(model, ssl_cfg)
    scheduler = build_scheduler(optimizer, ssl_cfg, ssl_epochs)

    print(f"\n  SSL pre-training: {ssl_epochs} epochs  |  {len(ds)} CUB images  "
          f"|  batch={ssl_batch_size}  T={ssl_temperature}")

    for epoch in range(ssl_epochs):
        loss = _ssl_epoch(model, loader, optimizer, device, ssl_temperature)
        print(f"  SSL {epoch+1:3d}/{ssl_epochs}  loss={loss:.4f}")
        if wandb_run:
            wandb_run.log({'ssl/epoch': epoch + 1, 'ssl/nce_loss': loss})
        if scheduler:
            scheduler.step()

    # unfreeze classifier for subsequent source training
    for p in model.classifier.parameters():
        p.requires_grad_(True)
    if getattr(model, 'classifier2', None) is not None:
        for p in model.classifier2.parameters():
            p.requires_grad_(True)
