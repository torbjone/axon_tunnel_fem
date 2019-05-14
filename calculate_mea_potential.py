import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

results_folder = "results"

imem = np.load(join(results_folder, "axon_imem.npy"))
tvec = np.load(join(results_folder, "axon_tvec.npy"))
source_pos = np.load(join(results_folder, "source_pos.npy"))
num_sources = source_pos.shape[0]
num_tsteps = len(tvec)
mesh_coordinates = np.load(join(results_folder, "mesh_coordinates.npy"))

closest_idxs = np.ones(num_sources, dtype=int)
pos_errors = np.zeros(num_sources)
for s_idx, s_pos in enumerate(source_pos):
    closest_idxs[s_idx] = np.argmin(np.sum((mesh_coordinates - s_pos)**2, axis=1))
    pos_errors[s_idx] = np.sqrt(np.sum((mesh_coordinates[closest_idxs[s_idx]] - s_pos)**2))
    print(closest_idxs[s_idx], mesh_coordinates[closest_idxs[s_idx]], s_pos, pos_errors[s_idx])

elec_xs = np.load(join(results_folder, "electrodes.npy"))
elec_ys = np.zeros(len(elec_xs))
elec_zs = np.zeros(len(elec_xs))

num_elecs = len(elec_xs)
lead_field = np.zeros((num_elecs, num_sources))
for e_idx in range(num_elecs):
    print(e_idx)
    lead_field[e_idx, :] = np.load(join(results_folder, "phi_elec_{}.npy".format(e_idx)))[closest_idxs]

phi = np.dot(lead_field, imem)
print(phi.shape, tvec.shape)
print(phi)

phi_norm = np.max(np.abs(phi))
for elec in range(num_elecs)[::4]:
    plt.plot(tvec, phi[elec] / phi_norm + elec)

plt.show()