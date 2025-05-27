import os
import sys
import time
import numpy as np
from ase.io import read
from quests.descriptor import get_descriptors
from quests.entropy import delta_entropy

## model name options: ["MACE-MP(M)", "MatterSim", "ORBv2", "SevenNet", "MACE-MPA", "CHGNet", "eqV2(OMat)"]
model_name = sys.argv[1]

dataset_name = "mptrj"
k, cutoff = 32, 5.0
path = f'/data/christine/mlip-arena/simulations_100/{dataset_name}/{model_name}_MPTrj_sims_supercell_1000K'

print(f"model name is {model_name}")
structures = []
for traj_file in sorted(os.listdir(path)):
    traj = read(os.path.join(path, traj_file), index=':')
    every_other = traj[2:49:2]
    structures.extend(every_other)
        
print(f"len structures is {len(structures)}")
assert(len(structures) == 2304)

print("starting descriptors")
t4 = time.time()
atoms_desc = get_descriptors(structures, k=k, cutoff=cutoff)
t5 = time.time()
print(f"descriptors took {t5-t4:.2f} seconds")

print("Loading 0th descriptor")
x_desc_0 = np.load(f"/data/christine/quests/{dataset_name}/{dataset_name}_without_subset_descriptors_0.npy")

print("Loading 1st descriptor")
x_desc_1 = np.load(f"/data/christine/quests/{dataset_name}/{dataset_name}_without_subset_descriptors_1.npy")

print("Loading 2nd descriptor")
x_desc_2 = np.load(f"/data/christine/quests/{dataset_name}/{dataset_name}_without_subset_descriptors_2.npy")

print("Loading 3rd descriptor")
x_desc_3 = np.load(f"/data/christine/quests/{dataset_name}/{dataset_name}_without_subset_descriptors_3.npy")

x_desc = np.vstack((x_desc_0, x_desc_1, x_desc_2, x_desc_3))

print("starting entropy")
t4 = time.time()
dH = delta_entropy(atoms_desc, x_desc, h=0.015)
t5 = time.time()
print(f"dH took {(t5-t4) / 60 / 60:.3f} hours")

np.save(f'mptrj_traj_{model_name}_100_dH.npy', dH)
print(f"Saved dH to mptrj_traj_{model_name}_100_dH.npy")
