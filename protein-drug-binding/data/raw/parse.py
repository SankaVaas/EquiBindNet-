"""
Parse PDBbind index file with binding data parsing
"""
import pandas as pd
import re
import numpy as np

def parse_binding_data(binding_str):
    """
    Parse binding data string and convert to pKd
    Examples:
        "Kd=49uM" -> 4.31
        "Ki=0.43uM" -> 6.37
        "Ki=0.068nM" -> 10.17
    """
    # Remove spaces
    binding_str = binding_str.strip()
    
    # Skip problematic entries
    if '<' in binding_str or '>' in binding_str or '~' in binding_str:
        return None
    
    # Extract value and unit using regex
    match = re.match(r'(Kd|Ki|IC50)=([\d.]+)(nM|uM|mM|pM)', binding_str, re.IGNORECASE)
    
    if not match:
        return None
    
    binding_type = match.group(1)
    value = float(match.group(2))
    unit = match.group(3)
    
    # Convert to Molar
    if unit == 'pM':
        molar = value * 1e-12
    elif unit == 'nM':
        molar = value * 1e-9
    elif unit == 'uM':
        molar = value * 1e-6
    elif unit == 'mM':
        molar = value * 1e-3
    else:
        return None
    
    # Calculate pKd = -log10(Kd in Molar)
    if molar > 0:
        pkd = -np.log10(molar)
        return pkd
    else:
        return None

def parse_pdbbind_index(index_file):
    """Parse PDBbind index file"""
    data = []
    
    with open(index_file, 'r') as f:
        for line in f:
            # Skip comments and headers
            if line.startswith('#') or line.strip() == '':
                continue
            
            parts = line.split()
            
            if len(parts) < 4:
                continue
            
            pdb_id = parts[0]
            resolution_str = parts[1]
            year_str = parts[2]
            binding_str = parts[3]
            
            # Parse resolution (skip NMR)
            if resolution_str == 'NMR':
                continue
            
            try:
                resolution = float(resolution_str)
                year = int(year_str)
            except:
                continue
            
            # Parse binding data
            pkd = parse_binding_data(binding_str)
            
            if pkd is None:
                continue
            
            data.append({
                'pdb_id': pdb_id,
                'resolution': resolution,
                'year': year,
                'pKd': pkd,
                'binding_raw': binding_str
            })
    
    return pd.DataFrame(data)


# Main execution
print("="*60)
print("PARSING PDBBIND INDEX FILE")
print("="*60)

df = parse_pdbbind_index('data/raw/index/INDEX_general_PL.2020R1.lst')

print(f"\n✅ Parsed {len(df)} valid complexes")
print(f"\nFirst 10 entries:")
print(df.head(10)[['pdb_id', 'pKd', 'resolution', 'year', 'binding_raw']])

print(f"\n📊 Statistics:")
print(f"   pKd range: {df['pKd'].min():.2f} - {df['pKd'].max():.2f}")
print(f"   Mean pKd: {df['pKd'].mean():.2f}")
print(f"   Median pKd: {df['pKd'].median():.2f}")
print(f"   Resolution range: {df['resolution'].min():.2f} - {df['resolution'].max():.2f} Å")
print(f"   Year range: {df['year'].min()} - {df['year'].max()}")

# Filter high quality
print("\n" + "="*60)
print("FILTERING HIGH-QUALITY COMPLEXES")
print("="*60)

high_quality = df[
    (df['resolution'] <= 2.5) &   # Good crystal structure
    (df['pKd'] >= 4.0) &           # At least micromolar binding
    (df['pKd'] <= 12.0) &          # Exclude unrealistic values
    (df['year'] >= 2000)           # Modern structures
].copy()

print(f"✅ High-quality subset: {len(high_quality)} complexes")

# Sort by pKd (strongest binders first)
high_quality = high_quality.sort_values('pKd', ascending=False).reset_index(drop=True)

print(f"\nTop 10 strongest binders:")
print(high_quality.head(10)[['pdb_id', 'pKd', 'resolution', 'binding_raw']])

# Create datasets of different sizes
print("\n" + "="*60)
print("CREATING DATASET FILES")
print("="*60)

# Tiny: 10 samples (quick testing)
tiny = high_quality.head(10)
tiny.to_csv('dataset_tiny.csv', index=False)
print(f"✅ dataset_tiny.csv - {len(tiny)} samples")

# Small: 50 samples (CPU training today)
small = high_quality.head(50)
small.to_csv('dataset_small.csv', index=False)
print(f"✅ dataset_small.csv - {len(small)} samples")

# Medium: 200 samples (better results)
if len(high_quality) >= 200:
    medium = high_quality.head(200)
    medium.to_csv('dataset_medium.csv', index=False)
    print(f"✅ dataset_medium.csv - {len(medium)} samples")

# Full: all high-quality
high_quality.to_csv('dataset_full.csv', index=False)
print(f"✅ dataset_full.csv - {len(high_quality)} samples")

print("\n" + "="*60)
print("RECOMMENDATION FOR TODAY")
print("="*60)
print("Start with: dataset_small.csv (50 samples)")
print("   - Trains in ~2 hours on CPU")
print("   - Good for proof-of-concept")
print("   - Can scale up later")

print("\n✅ Ready to download structures!")
print("   Run: python download_structures.py")