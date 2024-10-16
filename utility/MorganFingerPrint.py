import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pandas as pd
import os
import numpy as np

# Function to generate Morgan fingerprint and bit info
def generate_morgan_fingerprint(mol, radius=2, nBits=1024):
    bit_info = {}
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, bitInfo=bit_info)
    return morgan_fp, bit_info

# Function to generate and save substructure image
def save_substructure_image(mol, bit, bit_info, directory):
    atom_idx, radius = bit_info[bit][0]
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    amap = {}
    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
    
    img = Draw.MolToImage(submol, size=(300, 300))
    img_path = os.path.join(directory, f'bit_{bit}.png')
    img.save(img_path)
    return img_path

# Read the SDF file
sdf_file = "preprocs.sdf"
suppl = Chem.SDMolSupplier(sdf_file)

# Create directories to store the images and results
os.makedirs('substructure_images', exist_ok=True)
os.makedirs('results', exist_ok=True)

# List to store fingerprints for all molecules
all_fingerprints = []
results = []
results_bit = []
bits_smiles = {}

# Process each molecule in the SDF file
for idx, mol in enumerate(suppl):
    if mol is not None:
        # Generate Morgan fingerprint
        morgan_fp, bit_info = generate_morgan_fingerprint(mol)
        
        # Convert fingerprint to numpy array and add to list
        fp_array = np.array(morgan_fp)
        all_fingerprints.append(fp_array)
        
        # Process substructures for the first molecule only
        for bit in range(1024):
            if bit in bit_info and bit not in results_bit:
                if bit not in results_bit:
                    atom_idx, radius = bit_info[bit][0]
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                    amap = {}
                    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                    smiles = Chem.MolToSmiles(submol)
                    
                    # Generate and save substructure image
                    img_path = save_substructure_image(mol, bit, bit_info, 'substructure_images')
                    
                    results.append({
                        'Bit': bit,
                        'SMILES': smiles,
                        'Substructure_Image': img_path
                    })

                    results_bit.append(bit)
                    bits_smiles[bit] = smiles
                
                else:
                    if bits_smiles[bit] != smiles:
                        print('difference bit smiles !! molecule idx : ', idx, ', bit : ', bit, ', bit smiles : ', bits_smiles[bit], ', molecule_smiles: ', smiles)
            
# Save substructure info for the first molecule
df_substructures = pd.DataFrame(results)
df_substructures.to_csv('./data/morgan_fingerprint_substructures.csv', index=False)
print("Substructure results saved to results/morgan_fingerprint_substructures.csv")

# Convert list of fingerprints to DataFrame
df_fingerprints = pd.DataFrame(all_fingerprints)

# Add column names (Bit0, Bit1, ..., Bit1023)
df_fingerprints.columns = [f'Bit{i}' for i in range(1024)]

# Save all fingerprints to a single CSV file
df_fingerprints.to_csv('./data/all_molecule_fingerprints.csv', index=False)
print("All molecule fingerprints saved to results/all_molecule_fingerprints.csv")

print("Substructure images saved in the 'substructure_images' directory")