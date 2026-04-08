"""
Main training script for wafer defect classification.

New Architecture:
- Backbone: DINOv3 (frozen)
- Classification: Gate + Fine with shared feature tower
- Anomaly Detection: Dinomaly2 (Loose Reconstruction)

Usage:
    # Real data with DINOv3
    PYTHONPATH=. python wafer_defect/train.py --data_dir /path/to/wafer_data --use_dinov3 --epochs 10

    # Synthetic data (for testing)
    PYTHONPATH=. python wafer_defect/train.py --synthetic --epochs 10 --num_samples 200

    # Three-view fusion mode
    PYTHONPATH=. python wafer_defect/train.py --data_dir /path/to/wafer_data --use_dinov3 --three_views
"""

import argparse
import os
import platform
import torch

from wafer_defect.data.dataset import (
    generate_synthetic_dataset,
    create_dataloaders,
    create_real_dataloaders,
)
from wafer_defect.models import WaferDefectModel, WaferDefectModelSimple
from wafer_defect.engine.trainer import WaferDefectTrainer, generate_markdown_report


def parse_args():
    parser = argparse.ArgumentParser(description="Wafer Defect Classification (New Architecture)")

    # ─────────────────────────────────────────────────────────────────────────
    # Data source
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (default: use --data_dir)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to real data folder")

    # ─────────────────────────────────────────────────────────────────────────
    # Model architecture
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument("--use_dinov3", type=bool, default=True,
                        help="Use DINOv3 backbone (default: True)")
    parser.add_argument("--no_dinov3", action="store_true",
                        help="Disable DINOv3, use simple CNN instead")
    parser.add_argument("--backbone", type=str, default="dinov3_vitl16",
                        help="Backbone name (default: dinov3_vitl16)")
    parser.add_argument("--pretrained_path", type=str,
                        default="dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                        help="Path to pretrained weights")
    parser.add_argument("--three_views", action="store_true",
                        help="Enable 3-view fusion mode (requires I00K/I01K/I02K naming)")

    # ─────────────────────────────────────────────────────────────────────────
    # Dinomaly anomaly detection (open-source)
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument("--use_dinomaly", type=bool, default=True,
                        help="Use Dinomaly (open-source) anomaly detection (default: True)")
    parser.add_argument("--no_dinomaly", action="store_true",
                        help="Disable Dinomaly")
    parser.add_argument("--dinomaly_iters", type=int, default=10000,
                        help="Training iterations for Dinomaly decoder")
    parser.add_argument("--dinomaly_lr", type=float, default=2e-3,
                        help="Learning rate for Dinomaly decoder")
    parser.add_argument("--dinomaly_layers", nargs='+', type=int, default=[3, 4, 5, 6, 7, 8, 9, 10],
                        help="DINOv3 layer indices for Dinomaly")
    parser.add_argument("--dinomaly_dropout", type=float, default=0.2,
                        help="Dropout rate for Dinomaly")

    # ─────────────────────────────────────────────────────────────────────────
    # Data parameters
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Total samples for synthetic data")
    parser.add_argument("--num_defect_classes", type=int, default=10,
                        help="Number of defect classes")
    parser.add_argument("--nuisance_ratio", type=float, default=0.3,
                        help="Ratio of nuisance samples")
    parser.add_argument("--nuisance_name", type=str, default="Nuisance",
                        help="Folder name for nuisance class")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image resize size (use 392 for best quality with DINOv3)")
    parser.add_argument("--crop_bottom", type=int, default=40,
                        help="Bottom crop pixels (scale bar area)")

    # ─────────────────────────────────────────────────────────────────────────
    # Training hyperparameters
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--defect_weight", type=float, default=3.0,
                        help="Weight for defect class in Gate loss (penalty for missing defects)")
    parser.add_argument("--embed_dim", type=int, default=1024,
                        help="Feature dimension")

    # ─────────────────────────────────────────────────────────────────────────
    # Output
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory")

    return parser.parse_args()


def main():
    args = parse_args()

    # Handle --no_dinov3 and --no_dinomaly flags
    if args.no_dinov3:
        args.use_dinov3 = False
    if args.no_dinomaly:
        args.use_dinomaly = False

    # ─────────────────────────────────────────────────────────────────────────
    # Print configuration
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Wafer Defect Classification Training")
    print("Architecture: DINOv3 + Gate/Fine + Dinomaly")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Backbone: {'DINOv3' if args.use_dinov3 else 'Simple CNN'}")
    print(f"3-view fusion: {args.three_views}")
    print(f"Dinomaly: {args.use_dinomaly}")
    print(f"Data: {'Synthetic' if args.synthetic else f'Real ({args.data_dir})'}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Load data
    # ─────────────────────────────────────────────────────────────────────────
    if args.synthetic or args.data_dir is None:
        print("[1] Generating synthetic dataset...")
        train_samples, val_samples = generate_synthetic_dataset(
            num_samples=args.num_samples,
            num_defect_classes=args.num_defect_classes,
            nuisance_ratio=args.nuisance_ratio,
            seed=42,
        )
        print(f"    Train: {len(train_samples)}, Val: {len(val_samples)} samples")

        train_loader, val_loader = create_dataloaders(
            train_samples=train_samples,
            val_samples=val_samples,
            batch_size=args.batch_size,
            img_size=args.img_size,
        )
        num_defect_classes = args.num_defect_classes
        real_dataset = None
    else:
        # Windows: num_workers=0 to avoid fork issues
        num_workers = 0 if platform.system() == 'Windows' else 4
        print(f"[1] Loading real data from: {args.data_dir}")
        train_loader, val_loader = create_real_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=num_workers,
            img_size=args.img_size,
            crop_bottom=args.crop_bottom,
            nuisance_name=args.nuisance_name,
            use_three_views=args.three_views,
        )
        real_dataset = train_loader.dataset.parent
        num_defect_classes = real_dataset.num_classes - 1
        print(f"    Defect classes: {num_defect_classes}")

    # Get class names
    _ds = real_dataset if real_dataset else train_loader.dataset
    class_names = _ds.get_class_names() if hasattr(_ds, 'get_class_names') else None
    print(f"    Class names: {class_names}")

    # ─────────────────────────────────────────────────────────────────────────
    # Create model
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2] Creating model...")
    if args.use_dinov3:
        model = WaferDefectModel(
            num_defect_classes=num_defect_classes,
            backbone_name=args.backbone,
            pretrained_path=args.pretrained_path,
            embed_dim=args.embed_dim,
            defect_weight=args.defect_weight,
            use_dinomaly=args.use_dinomaly,
            dinomaly_config={
                'img_size': args.img_size,
                'layer_indices': args.dinomaly_layers,
                'lr': args.dinomaly_lr,
                'dropout': args.dinomaly_dropout,
                'iters': args.dinomaly_iters,
            },
        )
        print(f"    Dinomaly: {'enabled' if args.use_dinomaly else 'disabled'}")
        print(f"    Dinomaly layers: {args.dinomaly_layers}")
        print(f"    Dinomaly iters: {args.dinomaly_iters}")
    else:
        model = WaferDefectModelSimple(
            num_defect_classes=num_defect_classes,
            img_size=args.img_size,
            feat_dim=512,
        )

    model = model.to(args.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {num_params:,}")

    # ─────────────────────────────────────────────────────────────────────────
    # Create optimizer and trainer
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3] Setting up trainer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    trainer = WaferDefectTrainer(
        model=model,
        optimizer=optimizer,
        device=args.device,
        defect_weight=args.defect_weight,
        output_dir=args.output_dir,
        class_names=class_names,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4] Starting training...")
    best_val_loss = float('inf')
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 60)

        # Train
        train_results = trainer.train_epoch(train_loader, epoch)
        print(f"\nTrain Loss: {train_results['train_loss']:.4f}")
        print(f"Gate - Accuracy: {train_results['gate_metrics']['accuracy']:.4f}, "
              f"Defect Recall: {train_results['gate_metrics']['defect_recall']:.4f}")
        print(f"Fine - Accuracy: {train_results['fine_metrics']['accuracy']:.4f}, "
              f"Macro F1: {train_results['fine_metrics']['macro_f1']:.4f}")

        # Validate
        val_results = trainer.validate(val_loader)
        print(f"\nVal Loss: {val_results['val_loss']:.4f}")
        print(f"Gate - Accuracy: {val_results['gate_metrics']['accuracy']:.4f}, "
              f"Defect Recall: {val_results['gate_metrics']['defect_recall']:.4f}")
        print(f"Fine - Accuracy: {val_results['fine_metrics']['accuracy']:.4f}, "
              f"Macro F1: {val_results['fine_metrics']['macro_f1']:.4f}")

        # Save best model
        if val_results['val_loss'] < best_val_loss:
            best_val_loss = val_results['val_loss']
            best_epoch = epoch
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            trainer.save_checkpoint(ckpt_path, epoch)
            print(f"\nNew best model! Loss: {best_val_loss:.4f}")

        # Record history
        history.append({
            'epoch': epoch,
            'train_loss': train_results['train_loss'],
            'val_loss': val_results['val_loss'],
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Train Dinomaly decoder (Phase 2)
    # ─────────────────────────────────────────────────────────────────────────
    if args.use_dinov3 and args.use_dinomaly and hasattr(model, 'train_dinomaly'):
        print("\n" + "=" * 60)
        print("Phase 2: Training Dinomaly Decoder")
        print("=" * 60)

        # Load best model first
        best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
        if os.path.exists(best_ckpt_path):
            trainer.load_checkpoint(best_ckpt_path)
            print("Loaded best model for Dinomaly training...")

        # Collect defect images from train_loader
        print("Collecting defect images...")
        defect_images = []
        for batch in train_loader:
            imgs = batch['images']
            is_defect = batch['is_defect']
            defect_mask = is_defect == 1
            if defect_mask.sum() > 0:
                defect_images.append(imgs[defect_mask])
        defect_images = torch.cat(defect_images, dim=0)
        print(f"  Total defect images: {len(defect_images)}")

        # Save the checkpoint path
        dinomaly_path = os.path.join(args.output_dir, "dinomaly_decoder.pt")

        # Train Dinomaly with defect images
        model.train_dinomaly(
            train_images=defect_images,
            device=args.device,
            save_path=dinomaly_path,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Final report
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")

    # Load best and run final validation
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(best_ckpt_path):
        trainer.load_checkpoint(best_ckpt_path)

    final_val = trainer.validate(val_loader, save_errors=True)

    # Generate report
    dataset_info = {
        'train': len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else 0,
        'val': len(val_loader.dataset) if hasattr(val_loader.dataset, '__len__') else 0,
        'num_classes': num_defect_classes + 1,
        'defect_classes': num_defect_classes,
    }
    report_path = generate_markdown_report(
        val_results=final_val,
        history=history,
        class_names=class_names,
        dataset_info=dataset_info,
        output_dir=args.output_dir,
        prefix="validation_report",
    )
    print(f"Report saved: {report_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
