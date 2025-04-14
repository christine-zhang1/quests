import time
import numpy as np
from quests.entropy import delta_entropy, approx_delta_entropy

dataset_name = "oc20"

print("Loading 0th descriptor")
x_desc_0 = np.load(f"{dataset_name}/{dataset_name}_without_subset_descriptors_0.npy")

print("Loading 1st descriptor")
x_desc_1 = np.load(f"{dataset_name}/{dataset_name}_without_subset_descriptors_1.npy")

print("Loading 2nd descriptor")
x_desc_2 = np.load(f"{dataset_name}/{dataset_name}_without_subset_descriptors_2.npy")

print("Loading 3rd descriptor")
x_desc_3 = np.load(f"{dataset_name}/{dataset_name}_without_subset_descriptors_3.npy")

print("Loading 4th descriptor")
x_desc_4 = np.load(f"{dataset_name}/{dataset_name}_without_subset_descriptors_4.npy")

print("Loading y descriptor")
y_desc = np.load(f"{dataset_name}/{dataset_name}_subset_descriptors.npy")

x_desc = np.vstack((x_desc_0, x_desc_1, x_desc_2, x_desc_3, x_desc_4))

k, cutoff = 32, 5.0

print("computing entropy")

# computes dH (Y | X)
t4 = time.time()
dH = delta_entropy(y_desc, x_desc, h=0.015)
t5 = time.time()
print(f"dH took {(t5-t4) / 60 / 60:.3f} hours")

np.save(f'{dataset_name}/{dataset_name}_subset_dH.npy', dH)
print(f"Saved dH to {dataset_name}/{dataset_name}_subset_dH.npy")

# computes approximate dH (Y | X)
# t4 = time.time()
# dH_approx = approx_delta_entropy(y_desc, x_desc, h=0.015, n=5, graph_neighbors=10)
# t5 = time.time()
# print(f"approximate dH took {t5-t4:.2f} seconds")

# np.save('mptrj_subset_approximate_dH.npy', dH_approx)
# print(f"Saved approximate dH to mptrj_subset_approximate_dH.npy")
