"""
Test SE(3)-equivariance: predictions should be invariant to rotations
"""
import torch
import numpy as np
from models.equibind_model import EquiBindNet
from scipy.spatial.transform import Rotation

device = torch.device('cpu')

# Load trained model with CORRECT architecture
model = EquiBindNet(
    num_atom_types=10,
    hidden_dim=128,      # ✅ Changed from 64 to 128
    num_layers=4,        # ✅ Changed from 3 to 4
    use_simple=True
).to(device)

model.load_state_dict(torch.load('checkpoints/EquiBindNet_best.pth', weights_only=True))
model.eval()

# Load a test graph
import pandas as pd
test_df = pd.read_csv('data/processed/test.csv')
sample_graph = torch.load(test_df.iloc[0]['graph_file'], weights_only=False)

print("="*60)
print("TESTING SE(3)-EQUIVARIANCE")
print("="*60)
print(f"\nTest sample: {test_df.iloc[0]['pdb_id']}")
print(f"True pKd: {test_df.iloc[0]['pKd']:.2f}")

# Original prediction
with torch.no_grad():
    original_pred = model(sample_graph).item()

print(f"\nOriginal prediction: {original_pred:.4f}")

# Test multiple random rotations
print("\nApplying random rotations...")
predictions = [original_pred]

for i in range(10):
    # Generate random rotation
    rotation = Rotation.random().as_matrix()
    rotation_tensor = torch.tensor(rotation, dtype=torch.float32)
    
    # Rotate coordinates
    rotated_graph = sample_graph.clone()
    rotated_graph.pos = torch.mm(sample_graph.pos, rotation_tensor.T)
    
    # Predict
    with torch.no_grad():
        rotated_pred = model(rotated_graph).item()
    
    predictions.append(rotated_pred)
    print(f"  Rotation {i+1}: {rotated_pred:.4f} (diff: {abs(rotated_pred - original_pred):.6f})")

# Statistics
predictions = np.array(predictions)
mean_pred = predictions.mean()
std_pred = predictions.std()
max_diff = np.abs(predictions - original_pred).max()

print("\n" + "="*60)
print("EQUIVARIANCE TEST RESULTS")
print("="*60)
print(f"Mean prediction: {mean_pred:.4f}")
print(f"Std deviation: {std_pred:.6f}")
print(f"Max difference: {max_diff:.6f}")

if std_pred < 0.01:
    print("\n✅ PASS: Model is rotation-invariant!")
    print("   Predictions are consistent across rotations")
elif std_pred < 0.1:
    print("\n⚠️  PARTIAL: Model shows some rotation invariance")
    print("   Small variations may be due to numerical precision")
else:
    print("\n❌ FAIL: Model is NOT rotation-invariant")
    print("   Predictions vary significantly with rotations")

print(f"\n🎯 Relative error: {(std_pred / abs(mean_pred) * 100):.2f}%")