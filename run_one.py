"""
run_one.py — single experiment runner for Slurm array jobs

Usage:
  python run_one.py --config configs/ctop_sweep.json --idx 0
  python run_one.py --config configs/ptoc_sweep.json --idx $SLURM_ARRAY_TASK_ID
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import wandb
from torchvision.datasets import ImageFolder

from src.data import (
    get_transform, UnlabeledDataset, PseudoLabeledDataset,
    make_loader, set_seed,
)
from src.models import ResNetUDA
from src.training import (
    train_source_epoch, train_joint_epoch,
    evaluate, generate_pseudo_labels,
    build_optimizer, build_scheduler,
)
from src.shot import run_shot_adaptation


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULTS = {
    # paths
    'cub_path':       './CUB_200_2011/images',
    'paintings_path': './CUB_200_Paintings',
    'ckpt_dir':       './checkpoints',
    # model
    'proj_dim': 128,
    # training
    'optimizer':    'sgd',
    'lr':           0.01,
    'momentum':     0.9,
    'weight_decay': 5e-4,
    'scheduler':    'cosine',
    'batch_size':   64,
    'num_workers':  4,
    'seed':         42,
    # augmentation
    'aug_mode': 'none',
    # joint_pseudo warmup
    'pseudo_warmup':       30,
    'pseudo_rounds':        3,
    'pseudo_round_epochs': 10,
    'pseudo_threshold':   0.7,
    'lambda_nce':         0.1,
    'temperature':        0.1,
    'noise_std':          0.05,
    'lambda_em':          0.0,
    'mixup_alpha':        0.0,
    'label_smoothing':    0.0,
    # SHOT
    'shot_lr':       1e-4,
    'shot_rounds':      3,
    'shot_epochs':     15,
    'lambda_im':      1.0,
    'lambda_pseudo':  1.0,
    # pipeline
    'mode':       'joint_pseudo_then_shot',  # joint_pseudo | shot_only | joint_pseudo_then_shot
    'src_epochs':  80,                        # used by shot_only (no pseudo rounds)
    # wandb
    'wandb_project': 'uda-cub',
    '_global_step':  0,
}


def load_config(config_path: str, idx: int) -> dict:
    with open(config_path) as f:
        configs = json.load(f)
    if idx >= len(configs):
        raise IndexError(f"Config index {idx} out of range (len={len(configs)})")
    cfg = dict(DEFAULTS)
    cfg.update(configs[idx])
    return cfg


# ── Training pipeline ─────────────────────────────────────────────────────────

def run_joint_pseudo(model, cfg, device, src_loader, src_eval_loader,
                     tgt_loader, tgt_root, tgt_tf, eval_loader, wandb_run):
    """
    Warmup (joint CE+NCE) + pseudo-labeling rounds.
    Returns best target accuracy seen during training.
    """
    criterion  = nn.CrossEntropyLoss(label_smoothing=cfg.get('label_smoothing', 0.0))
    optimizer  = build_optimizer(model, cfg)
    scheduler  = build_scheduler(optimizer, cfg, cfg['pseudo_warmup'])
    source_ds  = src_loader.dataset
    step       = cfg['_global_step']
    best_tgt   = 0.0

    # ── Warmup ────────────────────────────────────────────────
    for epoch in range(cfg['pseudo_warmup']):
        ce, nce, em, acc = train_joint_epoch(
            model, src_loader, tgt_loader, optimizer, criterion, device,
            cfg['lambda_nce'], cfg['temperature'], cfg['noise_std'],
            cfg['lambda_em'], cfg['mixup_alpha'])
        src_acc = evaluate(model, src_eval_loader, device)
        tgt_acc = evaluate(model, eval_loader, device)
        best_tgt = max(best_tgt, tgt_acc)
        step += 1

        print(f"  Warmup {epoch+1}/{cfg['pseudo_warmup']}"
              f"  CE={ce:.4f}  NCE={nce:.4f}  src={src_acc:.2f}%  tgt={tgt_acc:.2f}%")

        if wandb_run:
            wandb_run.log({
                'warmup/epoch': epoch + 1,
                'warmup/ce_loss': ce, 'warmup/nce_loss': nce,
                'warmup/train_acc': acc, 'warmup/src_acc': src_acc,
                'warmup/tgt_acc': tgt_acc,
            }, step=step)

        if scheduler:
            scheduler.step()

    # ── Pseudo rounds ─────────────────────────────────────────
    for rnd in range(1, cfg['pseudo_rounds'] + 1):
        indices, pseudo_labels = generate_pseudo_labels(
            model, tgt_root, tgt_tf, device,
            cfg['pseudo_threshold'], cfg['batch_size'], cfg['num_workers'])
        n_pseudo = len(indices)
        pct      = 100 * n_pseudo / len(eval_loader.dataset)
        print(f"\n  [Round {rnd}/{cfg['pseudo_rounds']}]"
              f"  pseudo={n_pseudo} ({pct:.1f}%)  thr={cfg['pseudo_threshold']}")

        if n_pseudo == 0:
            print("  No pseudo-labels — skipping round.")
            continue

        pseudo_ds   = PseudoLabeledDataset(tgt_root, tgt_tf, indices, pseudo_labels)
        from torch.utils.data import ConcatDataset
        combined_ds = ConcatDataset([source_ds, pseudo_ds])
        combined_ldr = make_loader(combined_ds, cfg['batch_size'], shuffle=True,
                                    num_workers=cfg['num_workers'])

        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg, cfg['pseudo_round_epochs'])

        for epoch in range(cfg['pseudo_round_epochs']):
            ce, nce, em, acc = train_joint_epoch(
                model, combined_ldr, tgt_loader, optimizer, criterion, device,
                cfg['lambda_nce'], cfg['temperature'], cfg['noise_std'],
                cfg['lambda_em'], cfg['mixup_alpha'])
            src_acc = evaluate(model, src_eval_loader, device)
            tgt_acc = evaluate(model, eval_loader, device)
            best_tgt = max(best_tgt, tgt_acc)
            step += 1

            print(f"  R{rnd} ep{epoch+1}/{cfg['pseudo_round_epochs']}"
                  f"  CE={ce:.4f}  src={src_acc:.2f}%  tgt={tgt_acc:.2f}%")

            if wandb_run:
                wandb_run.log({
                    f'round{rnd}/epoch': epoch + 1,
                    f'round{rnd}/ce_loss': ce,
                    f'round{rnd}/src_acc': src_acc,
                    f'round{rnd}/tgt_acc': tgt_acc,
                    f'round{rnd}/pseudo_count': n_pseudo,
                }, step=step)

            if scheduler:
                scheduler.step()

    cfg['_global_step'] = step
    return best_tgt


def run_shot_only(model, cfg, device, src_loader, src_eval_loader,
                  tgt_root, tgt_tf, eval_loader, wandb_run):
    """Source-only CE training → SHOT adaptation."""
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get('label_smoothing', 0.0))
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, cfg['src_epochs'])
    step      = cfg['_global_step']

    for epoch in range(cfg['src_epochs']):
        loss, acc = train_source_epoch(model, src_loader, optimizer, criterion,
                                        device, cfg.get('mixup_alpha', 0.0))
        src_acc = evaluate(model, src_eval_loader, device)
        step += 1
        print(f"  Source {epoch+1}/{cfg['src_epochs']}"
              f"  loss={loss:.4f}  acc={acc:.4f}  src_eval={src_acc:.2f}%")
        if wandb_run:
            wandb_run.log({'source/epoch': epoch + 1, 'source/loss': loss,
                            'source/train_acc': acc, 'source/eval_acc': src_acc}, step=step)
        if scheduler:
            scheduler.step()

    cfg['_global_step'] = step
    return evaluate(model, eval_loader, device)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to sweep JSON file')
    parser.add_argument('--idx',    type=int, required=True, help='Config index (SLURM_ARRAY_TASK_ID)')
    args = parser.parse_args()

    cfg    = load_config(args.config, args.idx)
    set_seed(cfg['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice  : {device}")
    print(f"Setting : {cfg['setting']}  mode={cfg['mode']}")
    print(f"Config  : {cfg.get('name', f'idx_{args.idx}')}")
    print(json.dumps({k: v for k, v in cfg.items()
                      if k not in ('_global_step', 'cub_path', 'paintings_path')}, indent=2))

    # ── wandb init ────────────────────────────────────────────
    run = wandb.init(
        project=cfg.get('wandb_project', 'uda-cub'),
        name=f"{cfg['setting']}_{cfg.get('name', f'idx_{args.idx}')}",
        config={k: v for k, v in cfg.items() if not k.startswith('_')},
        tags=[cfg['setting'], cfg.get('mode', 'joint_pseudo')],
    )

    # ── Paths & transforms ────────────────────────────────────
    setting    = cfg['setting']
    src_domain = 'cub'       if setting == 'CtoP' else 'paintings'
    tgt_domain = 'paintings' if setting == 'CtoP' else 'cub'
    src_root   = cfg['cub_path']       if setting == 'CtoP' else cfg['paintings_path']
    tgt_root   = cfg['paintings_path'] if setting == 'CtoP' else cfg['cub_path']

    _heavy   = cfg['aug_mode'] == 'heavy'
    _augment = cfg['aug_mode'] in ('light', 'heavy')
    src_tf      = get_transform(src_domain, augment=_augment, heavy=_heavy)
    src_clean_tf = get_transform(src_domain)
    tgt_tf      = get_transform(tgt_domain)

    from torchvision.datasets import ImageFolder
    source_ds    = ImageFolder(root=src_root, transform=src_tf)
    src_eval_ds  = ImageFolder(root=src_root, transform=src_clean_tf)
    eval_ds      = ImageFolder(root=tgt_root, transform=tgt_tf)
    tgt_unl_ds   = UnlabeledDataset(ImageFolder(root=tgt_root, transform=tgt_tf))

    src_loader      = make_loader(source_ds,   cfg['batch_size'], shuffle=True,  num_workers=cfg['num_workers'])
    src_eval_loader = make_loader(src_eval_ds, cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
    eval_loader     = make_loader(eval_ds,     cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
    tgt_loader      = make_loader(tgt_unl_ds,  cfg['batch_size'], shuffle=True,  num_workers=cfg['num_workers'])

    # ── Model ─────────────────────────────────────────────────
    model = ResNetUDA(num_classes=200, proj_dim=cfg['proj_dim']).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params  : {n_params/1e6:.2f}M\n")

    best_tgt = 0.0

    # ── Run pipeline ──────────────────────────────────────────
    mode = cfg.get('mode', 'joint_pseudo_then_shot')

    if mode in ('joint_pseudo', 'joint_pseudo_then_shot'):
        print("=" * 60)
        print("  Phase 1: Joint Pseudo Training")
        print("=" * 60)
        best_tgt = run_joint_pseudo(
            model, cfg, device,
            src_loader, src_eval_loader, tgt_loader,
            tgt_root, tgt_tf, eval_loader, run)

    elif mode in ('shot_only',):
        print("=" * 60)
        print("  Phase 1: Source-only Training")
        print("=" * 60)
        best_tgt = run_shot_only(
            model, cfg, device,
            src_loader, src_eval_loader, tgt_root, tgt_tf, eval_loader, run)

    if mode in ('shot_only', 'joint_pseudo_then_shot'):
        print("\n" + "=" * 60)
        print("  Phase 2: SHOT Adaptation")
        print("=" * 60)
        shot_best = run_shot_adaptation(
            model, tgt_root, tgt_tf, eval_loader, src_eval_loader,
            cfg, device, wandb_run=run)
        best_tgt = max(best_tgt, shot_best)

    # ── Final eval & save ─────────────────────────────────────
    final_tgt = evaluate(model, eval_loader, device)
    run.summary['best_tgt_acc']  = best_tgt
    run.summary['final_tgt_acc'] = final_tgt
    print(f"\n  best_tgt={best_tgt:.2f}%  final_tgt={final_tgt:.2f}%")

    os.makedirs(cfg['ckpt_dir'], exist_ok=True)
    ckpt_name = f"{setting}_{cfg.get('name', f'idx_{args.idx}')}.pth"
    torch.save(model.state_dict(), os.path.join(cfg['ckpt_dir'], ckpt_name))
    print(f"  Checkpoint → {cfg['ckpt_dir']}/{ckpt_name}")

    wandb.finish()


if __name__ == '__main__':
    main()
