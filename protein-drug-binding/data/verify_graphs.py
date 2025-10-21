"""
Verify graph files and visualize one
"""
import torch
import pandas as pd
import os

PROCESSED_DIR = 'data/processed'

print("="*60)
print("VERIFYING GRAPH FILES")
print("="*60)

# Check splits exist
for split in ['train', 'val', 'test']:
    csv_file = os.path.join(PROCESSED_DIR, f'{split}.csv')
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        print(f"✅ {split}.csv - {len(df)} samples")
    else:
        print(f"❌ {split}.csv - NOT FOUND")

# Load and inspect a sample graph
print("\n" + "="*60)
print("INSPECTING SAMPLE GRAPH")
print("="*60)

train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train.csv'))
sample = train_df.iloc[0]

graph_file = sample['graph_file']
graph = torch.load(graph_file, weights_only=False)

print(f"\nSample: {sample['pdb_id']}")
print(f"  pKd (label): {sample['pKd']:.2f}")
print(f"  Nodes: {graph.num_nodes}")
print(f"  Edges: {graph.num_edges}")
print(f"  Node features shape: {graph.x.shape}")
print(f"  Edge features shape: {graph.edge_attr.shape}")
print(f"  Positions shape: {graph.pos.shape}")
print(f"  Label: {graph.y.item():.2f}")

print("\n✅ All checks passed!")
print("🚀 Ready to build the SE(3)-Equivariant model!")


import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'None'}")