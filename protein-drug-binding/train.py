"""
Training script for EquiBindNet
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import model
# import sys
# sys.path.append('models')
from models.equibind_model import EquiBindNet, BaselineCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")

# ============================================================
# DATA LOADING
# ============================================================

class BindingAffinityDataset:
    """Dataset for loading preprocessed graphs with augmentation"""
    
    def __init__(self, csv_file, augment=False):
        self.data_df = pd.read_csv(csv_file)
        self.augment = augment
        print(f"📊 Loaded {len(self.data_df)} samples from {csv_file}")
        
    def __len__(self):
        return len(self.data_df)
    
    def random_rotate(self, pos):
        """Apply random 3D rotation"""
        theta = torch.rand(1) * 2 * np.pi
        phi = torch.rand(1) * 2 * np.pi
        
        # Rotation matrices
        Rx = torch.tensor([[1, 0, 0],
                          [0, torch.cos(theta), -torch.sin(theta)],
                          [0, torch.sin(theta), torch.cos(theta)]])
        
        Ry = torch.tensor([[torch.cos(phi), 0, torch.sin(phi)],
                          [0, 1, 0],
                          [-torch.sin(phi), 0, torch.cos(phi)]])
        
        R = torch.mm(Rx, Ry)
        return torch.mm(pos, R)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        graph_file = row['graph_file']
        
        # Load graph (PyTorch 2.6+ compatibility)
        graph = torch.load(graph_file, weights_only=False)
        
        if self.augment:
            # Apply augmentations
            if torch.rand(1) > 0.5:
                graph.pos = self.random_rotate(graph.pos)
            
            # Add small random noise to positions
            graph.pos += torch.randn_like(graph.pos) * 0.01
            
            # Random edge dropout
            if torch.rand(1) > 0.5:
                edge_mask = torch.rand(graph.edge_index.size(1)) > 0.1
                graph.edge_index = graph.edge_index[:, edge_mask]
        
        return graph

# Load datasets
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

train_dataset = BindingAffinityDataset('data/processed/train.csv')
val_dataset = BindingAffinityDataset('data/processed/val.csv')
test_dataset = BindingAffinityDataset('data/processed/test.csv')

# Create data loaders
batch_size = 16  # Increased for better stability
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"\n✅ Data loaders created")
print(f"   Batch size: {batch_size}")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")

# ============================================================
# MODEL INITIALIZATION
# ============================================================

print("\n" + "="*60)
print("INITIALIZING MODELS")
print("="*60)

# Initialize models with matched capacity
hidden_dim = 128
num_layers = 4

# EquiBindNet (our model)
equibind_model = EquiBindNet(
    num_atom_types=10,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    use_simple=True  # CPU-optimized
).to(device)

# Baseline (for comparison)
baseline_model = BaselineCNN(
    num_atom_types=10,
    hidden_dim=hidden_dim,
    num_layers=num_layers
).to(device)

print(f"\n✅ EquiBindNet parameters: {sum(p.numel() for p in equibind_model.parameters()):,}")
print(f"✅ Baseline parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")

# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_epoch(model, loader, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    for batch in loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        pred = model(batch).squeeze()
        target = batch.y.squeeze()
        
        loss = criterion(pred, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(pred.detach().cpu().numpy())
        targets.extend(target.detach().cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    
    # Calculate metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    
    # Pearson correlation
    if len(predictions) > 1:
        corr = np.corrcoef(predictions, targets)[0, 1]
    else:
        corr = 0.0
    
    return avg_loss, rmse, mae, corr

def evaluate(model, loader, criterion):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            pred = model(batch).squeeze()
            target = batch.y.squeeze()
            
            loss = criterion(pred, target)
            
            total_loss += loss.item()
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    
    if len(predictions) > 1:
        corr = np.corrcoef(predictions, targets)[0, 1]
    else:
        corr = 0.0
    
    return avg_loss, rmse, mae, corr, predictions, targets

# ============================================================
# TRAINING LOOP
# ============================================================

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred, target):
        # MSE Loss
        mse_loss = self.mse(pred, target)
        
        # Correlation Loss
        vx = pred - pred.mean()
        vy = target - target.mean()
        corr = (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8)
        corr_loss = 1 - corr
        
        # Smoothness Loss (L1)
        smooth_loss = torch.abs(pred[1:] - pred[:-1]).mean()
        
        return mse_loss + self.alpha * corr_loss + self.beta * smooth_loss

def train_model(model, model_name, num_epochs=100):
    """Complete training pipeline with improved training"""
    
    print("\n" + "="*60)
    print(f"TRAINING {model_name}")
    print("="*60)
    
    # Improved optimizer settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=2e-4,  # Lower initial learning rate
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the restart interval after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Combined loss function
    criterion = CombinedLoss(alpha=0.1, beta=0.05)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_rmse': [], 'val_rmse': [],
        'train_corr': [], 'val_corr': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_rmse, train_mae, train_corr = train_epoch(
            model, train_loader, optimizer, criterion
        )
        
        # Validate
        val_loss, val_rmse, val_mae, val_corr, _, _ = evaluate(
            model, val_loader, criterion
        )
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        history['train_corr'].append(train_corr)
        history['val_corr'].append(val_corr)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'checkpoints/{model_name}_best.pth')
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, RMSE: {train_rmse:.4f}, Corr: {train_corr:.3f}")
            print(f"  Val   - Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}, Corr: {val_corr:.3f}")
        
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f'checkpoints/{model_name}_best.pth', weights_only=True))
    
    return history

# Create checkpoint directory
os.makedirs('checkpoints', exist_ok=True)

# Update dataset with augmentation for training
train_dataset.augment = True

# Train both models with increased epochs
equibind_history = train_model(equibind_model, "EquiBindNet", num_epochs=100)
baseline_history = train_model(baseline_model, "Baseline", num_epochs=100)

# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "="*60)
print("FINAL EVALUATION ON TEST SET")
print("="*60)

criterion = nn.MSELoss()

# EquiBindNet
eq_test_loss, eq_test_rmse, eq_test_mae, eq_test_corr, eq_preds, eq_targets = evaluate(
    equibind_model, test_loader, criterion
)

print(f"\n✅ EquiBindNet:")
print(f"   Test RMSE: {eq_test_rmse:.4f}")
print(f"   Test MAE: {eq_test_mae:.4f}")
print(f"   Test Correlation: {eq_test_corr:.3f}")

# Baseline
bl_test_loss, bl_test_rmse, bl_test_mae, bl_test_corr, bl_preds, bl_targets = evaluate(
    baseline_model, test_loader, criterion
)

print(f"\n✅ Baseline:")
print(f"   Test RMSE: {bl_test_rmse:.4f}")
print(f"   Test MAE: {bl_test_mae:.4f}")
print(f"   Test Correlation: {bl_test_corr:.3f}")

# Comparison
improvement = ((bl_test_rmse - eq_test_rmse) / bl_test_rmse) * 100
print(f"\n🎯 Improvement: {improvement:.1f}% {'better' if improvement > 0 else 'worse'}")

# ============================================================
# VISUALIZATION
# ============================================================

print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('EquiBindNet vs Baseline Comparison', fontsize=16, fontweight='bold')

# Training curves
axes[0, 0].plot(equibind_history['train_loss'], label='EquiBindNet Train', linewidth=2)
axes[0, 0].plot(equibind_history['val_loss'], label='EquiBindNet Val', linewidth=2)
axes[0, 0].plot(baseline_history['train_loss'], label='Baseline Train', linewidth=2, linestyle='--')
axes[0, 0].plot(baseline_history['val_loss'], label='Baseline Val', linewidth=2, linestyle='--')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# RMSE
axes[0, 1].plot(equibind_history['val_rmse'], label='EquiBindNet', linewidth=2)
axes[0, 1].plot(baseline_history['val_rmse'], label='Baseline', linewidth=2, linestyle='--')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('RMSE')
axes[0, 1].set_title('Validation RMSE')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Correlation
axes[0, 2].plot(equibind_history['val_corr'], label='EquiBindNet', linewidth=2)
axes[0, 2].plot(baseline_history['val_corr'], label='Baseline', linewidth=2, linestyle='--')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Pearson Correlation')
axes[0, 2].set_title('Validation Correlation')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Predictions - EquiBindNet
axes[1, 0].scatter(eq_targets, eq_preds, alpha=0.6, s=50)
axes[1, 0].plot([eq_targets.min(), eq_targets.max()], 
                [eq_targets.min(), eq_targets.max()], 
                'r--', linewidth=2)
axes[1, 0].set_xlabel('True pKd')
axes[1, 0].set_ylabel('Predicted pKd')
axes[1, 0].set_title(f'EquiBindNet (R={eq_test_corr:.3f})')
axes[1, 0].grid(alpha=0.3)

# Predictions - Baseline
axes[1, 1].scatter(bl_targets, bl_preds, alpha=0.6, s=50, color='orange')
axes[1, 1].plot([bl_targets.min(), bl_targets.max()], 
                [bl_targets.min(), bl_targets.max()], 
                'r--', linewidth=2)
axes[1, 1].set_xlabel('True pKd')
axes[1, 1].set_ylabel('Predicted pKd')
axes[1, 1].set_title(f'Baseline (R={bl_test_corr:.3f})')
axes[1, 1].grid(alpha=0.3)

# Error distribution
eq_errors = np.abs(eq_preds - eq_targets)
bl_errors = np.abs(bl_preds - bl_targets)
axes[1, 2].hist(eq_errors, bins=15, alpha=0.6, label='EquiBindNet', color='blue')
axes[1, 2].hist(bl_errors, bins=15, alpha=0.6, label='Baseline', color='orange')
axes[1, 2].set_xlabel('Absolute Error')
axes[1, 2].set_ylabel('Count')
axes[1, 2].set_title('Error Distribution')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_results.png', dpi=150, bbox_inches='tight')
print("✅ Saved: results/training_results.png")

plt.show()

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)