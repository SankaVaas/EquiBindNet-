"""
Create train/validation/test splits
"""
import pandas as pd
import numpy as np
import os
import shutil

PROCESSED_DIR = 'data/processed'
DATASET = os.path.join(PROCESSED_DIR, 'dataset_processed.csv')

print("="*60)
print("CREATING DATA SPLITS")
print("="*60)

# Load processed dataset
df = pd.read_csv(DATASET)
print(f"\nTotal samples: {len(df)}")

# Set random seed
np.random.seed(42)

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Split: 70% train, 15% val, 15% test
n = len(df)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train_df = df[:train_end]
val_df = df[train_end:val_end]
test_df = df[val_end:]

print(f"\n📊 Split sizes:")
print(f"   Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
print(f"   Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
print(f"   Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")

# Save splits
train_df.to_csv(os.path.join(PROCESSED_DIR, 'train.csv'), index=False)
val_df.to_csv(os.path.join(PROCESSED_DIR, 'val.csv'), index=False)
test_df.to_csv(os.path.join(PROCESSED_DIR, 'test.csv'), index=False)

print(f"\n📊 pKd distribution:")
print(f"   Train: {train_df['pKd'].mean():.2f} ± {train_df['pKd'].std():.2f}")
print(f"   Val:   {val_df['pKd'].mean():.2f} ± {val_df['pKd'].std():.2f}")
print(f"   Test:  {test_df['pKd'].mean():.2f} ± {test_df['pKd'].std():.2f}")

print("\n✅ Splits created!")
print(f"   {PROCESSED_DIR}/train.csv")
print(f"   {PROCESSED_DIR}/val.csv")
print(f"   {PROCESSED_DIR}/test.csv")