"""
Download PDB structures from RCSB for selected dataset
"""
import pandas as pd
import urllib.request
import os
from time import sleep
import sys

def download_pdb(pdb_id, output_dir='structures'):
    """Download single PDB file from RCSB"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    # Skip if already exists
    if os.path.exists(output_file):
        size_kb = os.path.getsize(output_file) / 1024
        if size_kb > 1:  # Valid file
            return True, "exists", size_kb
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        urllib.request.urlretrieve(url, output_file)
        size_kb = os.path.getsize(output_file) / 1024
        return True, "downloaded", size_kb
    except Exception as e:
        return False, str(e), 0

# ============================================================
# CONFIGURATION
# ============================================================
DATASET = 'dataset_small.csv'  # Change to 'dataset_medium.csv' for more data
OUTPUT_DIR = 'structures'

print("="*60)
print("DOWNLOADING PDB STRUCTURES FROM RCSB")
print("="*60)

# Read dataset
df = pd.read_csv(DATASET)
print(f"\n📁 Dataset: {DATASET}")
print(f"📊 Total structures: {len(df)}")
print(f"🎯 Output directory: {OUTPUT_DIR}/")

print(f"\n⏱️  Estimated time: ~{len(df) * 0.5 / 60:.1f} minutes")
print("="*60)

# Download
success_count = 0
exists_count = 0
failed_count = 0
failed_list = []
total_size_mb = 0

for idx, row in df.iterrows():
    pdb_id = row['pdb_id']
    pkd = row['pKd']
    
    status, msg, size_kb = download_pdb(pdb_id, OUTPUT_DIR)
    
    if status:
        total_size_mb += size_kb / 1024
        if msg == "exists":
            exists_count += 1
            print(f"[{idx+1}/{len(df)}] ⏭️  {pdb_id} (pKd={pkd:.1f}) - already exists ({size_kb:.1f} KB)")
        else:
            success_count += 1
            print(f"[{idx+1}/{len(df)}] ✅ {pdb_id} (pKd={pkd:.1f}) - downloaded ({size_kb:.1f} KB)")
            sleep(0.5)  # Be polite to RCSB servers
    else:
        failed_count += 1
        failed_list.append(pdb_id)
        print(f"[{idx+1}/{len(df)}] ❌ {pdb_id} - FAILED: {msg}")

# Summary
print("\n" + "="*60)
print("DOWNLOAD COMPLETE")
print("="*60)
print(f"✅ Newly downloaded: {success_count}")
print(f"⏭️  Already existed: {exists_count}")
print(f"❌ Failed: {failed_count}")
print(f"📦 Total size: {total_size_mb:.1f} MB")

if failed_count > 0:
    print(f"\n⚠️  Failed PDB IDs: {failed_list}")
    print("   These might be obsolete or removed from RCSB")
    print("   You can:")
    print("   1. Continue with downloaded ones")
    print("   2. Remove failed IDs from CSV")
    print("   3. Try downloading manually")
else:
    print("\n🎉 All structures downloaded successfully!")

print(f"\n📁 Location: {OUTPUT_DIR}/")
print(f"✅ Ready for graph building!")