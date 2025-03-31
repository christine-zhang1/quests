# get a subset of mptrj
import os
import random
from ase.io import read, write

random.seed(88)

def random_sample_from_directory(directory, sample_size):
    files = sorted(os.listdir(directory))  # Get all files
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]  # Filter only files
    sample = random.sample(files, min(sample_size, len(files)))  # Pick random sample
    return sample

# Example usage
data_dir = "/data/shared/MPTrj/original"
sample_size = 500  # Change to the desired number of files

random_files = random_sample_from_directory(data_dir, sample_size)
print(len(random_files))
print(random_files[0])

# save the subset to a file
structures_1 = []
for filename in random_files:
    if filename.endswith(".extxyz"):
        filepath = os.path.join(data_dir, filename)
        atoms = read(filepath, index=":")  # Load all frames in case of trajectory files
        structures_1.extend(atoms)  # Add structures to list
        
# Save to a single file (extxyz or traj format)
save_path = "mptrj_subset_500.extxyz"
write(save_path, structures_1)

print(f"Saved {len(structures_1)} structures to {save_path}")


# now doing the rest of the dataset without this subset

files = sorted(os.listdir(data_dir))

# Read all .extxyz files
structures = []
for filename in files:
    if filename.endswith(".extxyz") and filename not in random_files:
        filepath = os.path.join(data_dir, filename)
        atoms = read(filepath, index=":")  # Load all frames in case of trajectory files
        structures.extend(atoms)  # Add structures to list

# Save to a single file (extxyz or traj format)
save_path = "mptrj_without_subset.extxyz"
write(save_path, structures)

print(f"Saved {len(structures)} structures to {save_path}")
