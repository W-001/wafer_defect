"""
Main training script for wafer defect classification.

Usage:
    python train.py --use_dinov3  # Use DINOv3 backbone (slower but better)
    python train.py               # Use simple CNN backbone (faster for testing)
"""

import argparse
import torch

from wafer_defect.data.dataset import generate_synthetic_dataset, create_dataloaders
from wafer_defect.models import WaferDefectModel, WaferDefectModelSimple
from wafer_defect.engine.trainer import WaferDefectTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Wafer Defect Classification")

    # Model
    parser.add_argument("--use_dinov3", action="store_true", help="Use DINOv3 backbone")
    parser.add_argument("--backbone", type=str, default="dinov3_vitl16", help="Backbone name")
    parser.add_argument("--pretrained_path", type=str, default="dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                        help="Path to pretrained weights")
    parser.add_argument("--embed_dim", type=int, default=1024, help="Feature dimension")

    # Data
    parser.add_argument("--num_samples", type=int, default=200, help="Total samples for synthetic data")
    parser.add_argument("--num_defect_classes", type=int, default=10, help="Number of defect classes")
    parser.add_argument("--nuisance_ratio", type=float, default=0.3, help="Ratio of nuisance samples")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")

    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gate_weight", type=float, default=1.0, help="Gate loss weight")
    parser.add_argument("--fine_weight", type=float, default=0.5, help="Fine loss weight")
    parser.add_argument("--metric_weight", type=float, default=0.1, help="Metric loss weight")
    parser.add_argument("--defect_weight", type=float, default=3.0, help="Defect class weight")

    # Other
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Wafer Defect Classification Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Use DINOv3: {args.use_dinov3}")
    print(f"Samples: {args.num_samples}, Defect Classes: {args.num_defect_classes}")
    print()

    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    train_samples, val_samples = generate_synthetic_dataset(
        num_samples=args.num_samples,
        num_defect_classes=args.num_defect_classes,
        nuisance_ratio=args.nuisance_ratio,
        seed=42
    )
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_samples=train_samples,
        val_samples=val_samples,
        batch_size=args.batch_size,
        num_workers=0  # Use 0 for synthetic data
    )

    # Create model
    print("\nCreating model...")
    if args.use_dinov3:
        model = WaferDefectModel(
            num_defect_classes=args.num_defect_classes,
            backbone_name=args.backbone,
            pretrained_path=args.pretrained_path,
            embed_dim=args.embed_dim,
            defect_weight=args.defect_weight
        )
    else:
        model = WaferDefectModelSimple(
            num_defect_classes=args.num_defect_classes,
            img_size=224,
            feat_dim=512
        )
    model = model.to(args.device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    # Create trainer
    trainer = WaferDefectTrainer(
        model=model,
        optimizer=optimizer,
        device=args.device,
        gate_weight=args.gate_weight,
        fine_weight=args.fine_weight,
        metric_weight=args.metric_weight,
        defect_weight=args.defect_weight
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 60}")

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
              f"Defect Recall: {val_results['gate_metrics']['defect_recall']:.4f}, "
              f"Defect->Nuisance Rate: {val_results['gate_metrics']['defect_as_nuisance_rate']:.4f}")
        print(f"Fine - Accuracy: {val_results['fine_metrics']['accuracy']:.4f}, "
              f"Macro F1: {val_results['fine_metrics']['macro_f1']:.4f}")

        # Save best model
        if val_results['val_loss'] < best_val_loss:
            best_val_loss = val_results['val_loss']
            print(f"\nNew best model! Loss: {best_val_loss:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Inference example
    print("\n" + "=" * 60)
    print("Inference Example")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        # Get a batch
        sample_batch = next(iter(val_loader))
        images = sample_batch["images"].to(args.device)

        outputs = model(images)

        print(f"\nSample batch:")
        print(f"  Gate predictions (is_defect): {outputs['is_defect_pred'].cpu().tolist()}")
        print(f"  Gate probs: {outputs['gate_prob'][:, 1].cpu().tolist()}")
        print(f"  Fine predictions: {outputs['fine_pred'].cpu().tolist()}")
        print(f"  Fine probs: {outputs['fine_prob'].max(dim=1)[0].cpu().tolist()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
