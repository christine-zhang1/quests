import sys
import time
import numpy as np
from ase.io import read
from quests.descriptor import get_descriptors
from quests.entropy import delta_entropy, approx_delta_entropy

num = int(sys.argv[1])

structures = read("oc20/oc20_without_subset.extxyz", index=":")
dset_y = read("oc20/oc20_subset.extxyz", index=":")
print("done reading in structures")

dataset_name = 'oc20'

k, cutoff = 32, 5.0

if num == 0:
    t0 = time.time()
    y = get_descriptors(dset_y, k=k, cutoff=cutoff)
    t1 = time.time()
    print(f"y get descriptors took {t1-t0:.2f} seconds")

    np.save(f'{dataset_name}_subset_descriptors.npy', y)
    print(f"Saved y descriptors to {dataset_name}_subset_descriptors.npy")

t2 = time.time()
if num == 4:
    x = get_descriptors(structures[400510*4:], k=k, cutoff=cutoff)
else:
    x = get_descriptors(structures[400510*num:400510*(num+1)], k=k, cutoff=cutoff)
t3 = time.time()
print(f"x get descriptors took {(t3-t2) / 60 / 60:.3f} hours")

fname = f"{dataset_name}_without_subset_descriptors_{num}.npy"
np.save(fname, x)
print(f"Saved x descriptors to {fname}")

# # computes approximate dH (Y | X)
# t4 = time.time()
# dH_approx = approx_delta_entropy(y, x, h=0.015, n=5, graph_neighbors=10)
# t5 = time.time()
# print(f"approximate dH took {t5-t4:.2f} seconds")

# np.save('mptrj_subset_approximate_dH.npy', dH_approx)
# print(f"Saved approximate dH to mptrj_subset_approximate_dH.npy")

# # computes dH (Y | X)
# t4 = time.time()
# dH = delta_entropy(y, x, h=0.015)
# t5 = time.time()
# print(f"dH took {t5-t4:.2f} seconds")

# np.save('mptrj_subset_dH.npy', dH)
# print(f"Saved dH to mptrj_subset_dH.npy")
