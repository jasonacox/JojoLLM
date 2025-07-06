#!/usr/bin/python3
"""Regenerate plot for existing checkpoint with fixed logic"""

import torch
import sys
import os
from utils import MetricsTracker, PlotManager

def regenerate_checkpoint_plot(checkpoint_path=None):
    """Regenerate the plot for an existing checkpoint"""
    
    if checkpoint_path is None:
        if len(sys.argv) < 2:
            print("Usage: python regenerate_plot.py <checkpoint_path>")
            print("Example: python regenerate_plot.py models/story_best.pt")
            
            # Show available checkpoint files
            import glob
            checkpoint_files = glob.glob("models/*.pt")
            if checkpoint_files:
                print(f"\nAvailable checkpoint files:")
                for file in sorted(checkpoint_files)[:10]:  # Show first 10
                    size_mb = os.path.getsize(file) / (1024 * 1024)
                    print(f"  {file} ({size_mb:.1f} MB)")
                if len(checkpoint_files) > 10:
                    print(f"  ... and {len(checkpoint_files) - 10} more")
            else:
                print("\nNo checkpoint files found in models/ directory")
            return
        checkpoint_path = sys.argv[1]
    
    # Validate checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return
    
    # Generate plot path by replacing .pt with .png
    plot_path = checkpoint_path.replace('.pt', '_regenerated.png')
    
    print(f"Regenerating plot for: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'metrics' not in checkpoint:
            print("No metrics found in checkpoint")
            return
        
        # Recreate the metrics tracker from checkpoint data
        metrics = MetricsTracker()
        metrics.metrics = checkpoint['metrics']
        
        # Get dataset name for title
        config = checkpoint.get('config', {})
        dataset_name = config.get('data', {}).get('dataset_name', 'unknown')
        epoch = checkpoint.get('epoch', 0)
        
        # Extract base filename for title
        base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        title = f"Training Progress - {dataset_name} ({base_name}, Epoch {epoch+1})"
        
        print(f"Generating plot: {plot_path}")
        success = PlotManager.plot_training_curves(metrics, plot_path, title)
        
        if success:
            print(f"✅ Plot generated successfully: {plot_path}")
            
            # Show metrics summary
            train_losses = metrics.get_metric_values('train_loss_eval')
            if not train_losses:
                epoch_losses = (metrics.get_metric_values('train_loss_epoch') or 
                               metrics.get_metric_values('train_loss'))
                if epoch_losses and len(epoch_losses) <= 2:
                    batch_losses = metrics.get_metric_values('train_loss_batch')
                    if batch_losses:
                        smooth_interval = max(1, len(batch_losses) // 100)
                        train_losses = [batch_losses[i] for i in range(0, len(batch_losses), smooth_interval)]
                        print(f"📊 Used smoothed batch data: {len(train_losses)} points from {len(batch_losses)} batches")
                    else:
                        train_losses = epoch_losses
                        print(f"📊 Used epoch data: {len(train_losses)} points")
                else:
                    train_losses = epoch_losses or []
                    print(f"📊 Used epoch data: {len(train_losses)} points")
            else:
                print(f"📊 Used evaluation data: {len(train_losses)} points")
                
            val_losses = (metrics.get_metric_values('val_loss_eval') or 
                         metrics.get_metric_values('val_loss_epoch') or 
                         metrics.get_metric_values('val_loss') or [])
            print(f"📊 Validation data: {len(val_losses)} points")
            
            if train_losses:
                print(f"📈 Train loss range: {min(train_losses):.4f} to {max(train_losses):.4f}")
            if val_losses:
                print(f"📈 Val loss range: {min(val_losses):.4f} to {max(val_losses):.4f}")
        else:
            print("❌ Plot generation failed")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    regenerate_checkpoint_plot()
