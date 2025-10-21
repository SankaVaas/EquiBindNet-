"""
Build graph representations from PDB structures
"""
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
import os
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
DATASET = 'data/raw/dataset_small.csv'
STRUCTURE_DIR = 'data/raw/structures'
OUTPUT_DIR = 'data/processed'
POCKET_CUTOFF = 10.0  # Angstroms - how far from ligand to include
EDGE_CUTOFF = 5.0     # Angstroms - connect atoms within this distance

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_ligand_center(structure):
    """Find center of ligand (HETATM)"""
    ligand_coords = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # HETATM are non-standard residues (ligands)
                het_flag = residue.id[0]
                if het_flag != ' ':  # Not standard amino acid
                    # Skip water
                    if residue.resname in ['HOH', 'WAT']:
                        continue
                    
                    for atom in residue:
                        ligand_coords.append(atom.coord)
    
    if len(ligand_coords) == 0:
        return None
    
    return np.array(ligand_coords).mean(axis=0)


def extract_pocket_atoms(structure, ligand_center, cutoff=10.0):
    """
    Extract protein atoms near ligand (binding pocket)
    """
    pocket_atoms = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Only standard amino acids
                if residue.id[0] == ' ':
                    for atom in residue:
                        # Skip hydrogens
                        if atom.element == 'H':
                            continue
                        
                        coord = atom.coord
                        distance = np.linalg.norm(coord - ligand_center)
                        
                        if distance <= cutoff:
                            pocket_atoms.append({
                                'element': atom.element.strip(),
                                'coord': coord,
                                'residue': residue.resname,
                                'atom_name': atom.name
                            })
    
    return pocket_atoms


def atoms_to_graph(atoms, edge_cutoff=5.0):
    """
    Convert atom list to PyTorch Geometric graph
    """
    if len(atoms) == 0:
        return None
    
    # Atom type encoding
    atom_types = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 
                  'CL': 6, 'BR': 7, 'I': 8, 'Other': 9}
    
    # Node features: one-hot encoded atom type
    node_features = []
    coords = []
    
    for atom in atoms:
        element = atom['element'].upper()
        atom_idx = atom_types.get(element, atom_types['Other'])
        
        # One-hot vector
        one_hot = [0] * len(atom_types)
        one_hot[atom_idx] = 1
        
        node_features.append(one_hot)
        coords.append(atom['coord'])
    
    x = torch.tensor(node_features, dtype=torch.float)
    pos = torch.tensor(coords, dtype=torch.float)
    
    # Build edges (connect nearby atoms)
    edge_index = []
    edge_attr = []
    
    num_atoms = len(atoms)
    
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            
            if dist <= edge_cutoff:
                # Undirected graph (both directions)
                edge_index.append([i, j])
                edge_index.append([j, i])
                
                # Edge feature: distance
                edge_attr.append([dist])
                edge_attr.append([dist])
    
    if len(edge_index) == 0:
        # No edges - skip this graph
        return None
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create graph
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos
    )
    
    return data


# ============================================================
# MAIN PROCESSING
# ============================================================

print("="*60)
print("BUILDING GRAPH REPRESENTATIONS")
print("="*60)

# Load dataset
df = pd.read_csv(DATASET)
print(f"\n📊 Dataset: {DATASET}")
print(f"📦 Total samples: {len(df)}")
print(f"🎯 Pocket cutoff: {POCKET_CUTOFF} Å")
print(f"🔗 Edge cutoff: {EDGE_CUTOFF} Å")

parser = PDBParser(QUIET=True)

successful = []
failed = []

print("\n" + "="*60)
print("Processing structures...")
print("="*60)

for idx, row in df.iterrows():
    pdb_id = row['pdb_id']
    pkd = row['pKd']
    
    pdb_file = os.path.join(STRUCTURE_DIR, f"{pdb_id}.pdb")
    
    try:
        # Parse structure
        structure = parser.get_structure(pdb_id, pdb_file)
        
        # Extract ligand center
        ligand_center = extract_ligand_center(structure)
        
        if ligand_center is None:
            print(f"[{idx+1}/{len(df)}] ⚠️  {pdb_id} - No ligand found")
            failed.append(pdb_id)
            continue
        
        # Extract binding pocket
        pocket_atoms = extract_pocket_atoms(structure, ligand_center, POCKET_CUTOFF)
        
        if len(pocket_atoms) < 10:  # Too few atoms
            print(f"[{idx+1}/{len(df)}] ⚠️  {pdb_id} - Pocket too small ({len(pocket_atoms)} atoms)")
            failed.append(pdb_id)
            continue
        
        # Build graph
        graph = atoms_to_graph(pocket_atoms, EDGE_CUTOFF)
        
        if graph is None:
            print(f"[{idx+1}/{len(df)}] ⚠️  {pdb_id} - Graph building failed")
            failed.append(pdb_id)
            continue
        
        # Add label (pKd)
        graph.y = torch.tensor([pkd], dtype=torch.float)
        graph.pdb_id = pdb_id
        
        # Save graph
        output_file = os.path.join(OUTPUT_DIR, f"{pdb_id}.pt")
        torch.save(graph, output_file)
        
        successful.append({
            'pdb_id': pdb_id,
            'pKd': pkd,
            'num_nodes': graph.num_nodes,
            'num_edges': graph.num_edges,
            'graph_file': output_file
        })
        
        print(f"[{idx+1}/{len(df)}] ✅ {pdb_id} - {graph.num_nodes} nodes, {graph.num_edges} edges, pKd={pkd:.2f}")
        
    except Exception as e:
        print(f"[{idx+1}/{len(df)}] ❌ {pdb_id} - Error: {str(e)}")
        failed.append(pdb_id)

# Save successful dataset
success_df = pd.DataFrame(successful)
success_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_processed.csv'), index=False)

# Summary
print("\n" + "="*60)
print("GRAPH BUILDING COMPLETE")
print("="*60)
print(f"✅ Success: {len(successful)}/{len(df)}")
print(f"❌ Failed: {len(failed)}")

if len(successful) > 0:
    print(f"\n📊 Graph Statistics:")
    print(f"   Avg nodes: {success_df['num_nodes'].mean():.1f}")
    print(f"   Avg edges: {success_df['num_edges'].mean():.1f}")
    print(f"   Min nodes: {success_df['num_nodes'].min()}")
    print(f"   Max nodes: {success_df['num_nodes'].max()}")

if failed:
    print(f"\n⚠️  Failed PDB IDs: {failed[:10]}")
    if len(failed) > 10:
        print(f"   ... and {len(failed)-10} more")

print(f"\n💾 Processed graphs saved to: {OUTPUT_DIR}/")
print(f"📄 Dataset file: {OUTPUT_DIR}/dataset_processed.csv")
print("\n✅ Ready for model training!")