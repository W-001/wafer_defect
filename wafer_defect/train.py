"""
Main training script for wafer defect classification.

Usage:
    # Real data
    python train.py --data_dir /path/to/wafer_data --use_dinov3

    # Synthetic data (for testing)
    python train.py --synthetic --epochs 10 --num_samples 200
"""

import argparse
import os
import platform
import torch

from wafer_defect.data.dataset import (
    generate_synthetic_dataset, create_dataloaders, create_real_dataloaders
)
from wafer_defect.models import WaferDefectModel, WaferDefectModelSimple
from wafer_defect.engine.trainer import WaferDefectTrainer, generate_markdown_report


def parse_args():
    parser = argparse.ArgumentParser(description="Wafer Defect Classification")

    # Data source
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (default: use --data_dir)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to real data folder")

    # Model
    parser.add_argument("--use_dinov3", action="store_true", help="Use DINOv3 backbone")
    parser.add_argument("--backbone", type=str, default="dinov3_vitl16", help="Backbone name")
    parser.add_argument("--pretrained_path", type=str,
                        default="dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                        help="Path to pretrained weights (relative to project root)")
    parser.add_argument("--embed_dim", type=int, default=1024, help="Feature dimension")

    # RAD anomaly detection
    parser.add_argument("--use_rad_anomaly", action="store_true",
                        help="Use RAD (multi-layer patch-KNN) instead of class-center anomaly detection")
    parser.add_argument("--rad_layer_indices", nargs='+', type=int, default=[3, 6, 9, 11],
                        help="DINOv3 layer indices for RAD bank (0-based, e.g. 3 6 9 11)")
    parser.add_argument("--rad_k_image", type=int, default=5,
                        help="Top-K nearest neighbor images for RAD patch-KNN")
    parser.add_argument("--rad_bank_path", type=str, default=None,
                        help="Path to pre-built RAD memory bank .pth")

    # Data params
    parser.add_argument("--num_samples", type=int, default=200, help="Total samples for synthetic")
    parser.add_argument("--num_defect_classes", type=int, default=10, help="Number of defect classes")
    parser.add_argument("--nuisance_ratio", type=float, default=0.3, help="Ratio of nuisance samples")
    parser.add_argument("--nuisance_name", type=str, default="Nuisance",
                        help="Folder name for nuisance class")
    parser.add_argument("--img_size", type=int, default=224, help="Image resize size")
    parser.add_argument("--crop_bottom", type=int, default=40, help="Bottom crop pixels (scale bar)")

    # Training
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gate_weight", type=float, default=1.0, help="Gate loss weight")
    parser.add_argument("--fine_weight", type=float, default=0.5, help="Fine loss weight")
    parser.add_argument("--metric_weight", type=float, default=0.1, help="Metric loss weight")
    parser.add_argument("--defect_weight", type=float, default=3.0, help="Defect class weight")

    # Other
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
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
    print(f"Data: {'Synthetic' if args.synthetic else f'Real ({args.data_dir})'}")
    print()

    # Load data
    if args.synthetic or args.data_dir is None:
        print("Generating synthetic dataset...")
        train_samples, val_samples = generate_synthetic_dataset(
            num_samples=args.num_samples,
            num_defect_classes=args.num_defect_classes,
            nuisance_ratio=args.nuisance_ratio,
            seed=42
        )
        print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

        train_loader, val_loader = create_dataloaders(
            train_samples=train_samples,
            val_samples=val_samples,
            batch_size=args.batch_size,
            num_workers=0,
            crop_footer=True,
            footer_pixels=args.crop_bottom
        )

        num_defect_classes = args.num_defect_classes
        real_dataset = None
    else:
        # Windows: num_workers=0 to avoid fork issues; Linux servers: use 4
        real_num_workers = 0 if platform.system() == 'Windows' else 4
        print(f"Loading real data from: {args.data_dir}")
        train_loader, val_loader = create_real_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=real_num_workers,
            img_size=args.img_size,
            crop_bottom=args.crop_bottom,
            nuisance_name=args.nuisance_name
        )

        real_dataset = train_loader.dataset.parent
        num_defect_classes = real_dataset.num_classes - 1
        print(f"Defect classes: {num_defect_classes}")
        print(f"Class names: {real_dataset.get_class_names()}")

    # Create model
    print("\nCreating model...")
    if args.use_dinov3:
        model = WaferDefectModel(
            num_defect_classes=num_defect_classes,
            backbone_name=args.backbone,
            pretrained_path=args.pretrained_path,
            embed_dim=args.embed_dim,
            defect_weight=args.defect_weight,
            use_rad_anomaly=args.use_rad_anomaly,
            rad_layer_indices=args.rad_layer_indices,
            rad_k_image=args.rad_k_image,
            rad_bank_path=args.rad_bank_path,
        )
        if args.use_rad_anomaly:
            print(f"[RAD] Multi-layer patch-KNN anomaly detection enabled")
            print(f"[RAD] Layer indices: {args.rad_layer_indices}")
            print(f"[RAD] K nearest images: {args.rad_k_image}")
    else:
        model = WaferDefectModelSimple(
            num_defect_classes=num_defect_classes,
            img_size=args.img_size,
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
        defect_weight=args.defect_weight,
        output_dir=args.output_dir,
        class_names=real_dataset.get_class_names() if real_dataset else None
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    best_epoch = 0
    history = []

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

        # Show misclassification summary
        summary = val_results.get('misclassification_summary')
        if summary:
            print(f"\n[Misclassification]")
            print(f"  Gate errors: {summary['gate_total_errors']} "
                  f"(漏检: {summary['gate_errors_by_type']['defect_as_nuisance']}, "
                  f"误报: {summary['gate_errors_by_type']['nuisance_as_defect']})")
            print(f"  Fine errors: {summary['fine_total_errors']}")

        # Record history
        epoch_record = {
            'epoch': epoch,
            'train_loss': train_results['train_loss'],
            'val_loss': val_results['val_loss'],
            'gate_accuracy': train_results['gate_metrics']['accuracy'],
            'gate_defect_recall': train_results['gate_metrics']['defect_recall'],
            'fine_accuracy': train_results['fine_metrics']['accuracy'],
            'fine_macro_f1': train_results['fine_metrics']['macro_f1'],
            'gate_errors': summary['gate_total_errors'] if summary else 0,
            'fine_errors': summary['fine_total_errors'] if summary else 0,
        }
        history.append(epoch_record)

        # Save best model
        if val_results['val_loss'] < best_val_loss:
            best_val_loss = val_results['val_loss']
            best_epoch = epoch
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            trainer.save_checkpoint(ckpt_path, epoch, extra={'history': history})
            print(f"\nNew best model! Loss: {best_val_loss:.4f} → saved to {ckpt_path}")

    # Save last checkpoint
    last_ckpt_path = os.path.join(args.output_dir, "last_model.pt")
    trainer.save_checkpoint(last_ckpt_path, args.epochs)
    print(f"\nLast model saved to {last_ckpt_path}")
    print(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")

    # Build RAD memory bank after training (RAD mode only)
    if model.use_rad_anomaly:
        print("\n" + "=" * 60)
        print("Building RAD Memory Bank")
        print("=" * 60)
        # Filter training loader to defect samples only for bank building
        # We use the full training loader since RADAnomalyHead checks defect_type internally
        bank_path = os.path.join(args.output_dir, "rad_bank.pth")
        model.build_rad_bank(train_loader, device=args.device, save_path=bank_path)

    # Calibrate anomaly threshold after training
    if hasattr(model, 'calibrate_anomaly_threshold'):
        print("\n" + "=" * 60)
        print("Calibrating Unknown Defect Detection Threshold")
        print("=" * 60)
        model.calibrate_anomaly_threshold(train_loader, device=args.device)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Reload best model and run final validation for report
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(best_ckpt_path):
        trainer.load_checkpoint(best_ckpt_path)
        print(f"\nLoaded best model (epoch {best_epoch}) for final report...")
        final_val = trainer.validate(val_loader, save_errors=True)
    else:
        final_val = val_results  # fallback to last epoch

    # Generate markdown report
    dataset_info = {
        'total': (len(train_loader.dataset) + len(val_loader.dataset)
                  if hasattr(train_loader.dataset, '__len__') else 0),
        'train': len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else 0,
        'val': len(val_loader.dataset) if hasattr(val_loader.dataset, '__len__') else 0,
        'num_classes': num_defect_classes + 1,
        'defect_classes': num_defect_classes,
    }
    report_path = generate_markdown_report(
        val_results=final_val,
        history=history,
        class_names=(
            real_dataset.get_class_names() if real_dataset else
            list(range(num_defect_classes + 1))
        ),
        dataset_info=dataset_info,
        output_dir=args.output_dir,
        prefix="validation_report"
    )

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

        if outputs.get('is_unknown_defect') is not None:
            unknown = outputs['is_unknown_defect'].cpu().tolist()
            print(f"  Unknown defect flags: {unknown}")
            print(f"    (1 = novel/unknown defect, 0 = known defect type)")

        if outputs.get('anomaly_score') is not None:
            print(f"  Anomaly scores: {[f'{s:.4f}' for s in outputs['anomaly_score'].cpu().tolist()]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
