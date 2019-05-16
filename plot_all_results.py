import os
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import dolfin as df

out_folder = 'results'
sim_name = "tunnel_test"
fem_fig_folder = "fem_figs"
[os.makedirs(f, exist_ok=True) for f in [out_folder, fem_fig_folder]]

source_pos = np.load(join(out_folder, "source_pos.npy"))
imem = np.load(join(out_folder, "axon_imem.npy"))
num_tsteps = imem.shape[1]
num_sources = source_pos.shape[0]

def plot_all_results():
    """ Plot the set-up and potential at xz-plane,
    as well as the potential in the center along the z-axis (depth)
    """
    plt.close("all")
    fig = plt.figure(figsize=[18, 9])
    # fig.suptitle('Charge positions:\n%s' % (str(charge_pos)))
    fig.subplots_adjust(wspace=0.5, bottom=0.25, top=0.85)
    # ax1 = fig.add_subplot(131, aspect=1, ylabel='z [mm]', xlabel='x [mm]', title='Set up')
    ax2 = fig.add_subplot(211, aspect=1, ylabel='z [mm]', xlabel='x [mm]',
                          title='Field cross section')
    ax3 = fig.add_subplot(212, xlabel='x [mm]', ylabel='Field strength', ylim=[-0.6, 0.6])

    t_idx = 0

    df.File(join(out_folder, "phi_t_vec_{}.xml".format(t_idx))) >> phi


    mea_x_values = np.zeros(len(x))
    # analytic = np.zeros(len(x))
    for idx in range(len(x)):
        mea_x_values[idx] = phi(x[idx], 0, eps)
        # analytic[idx] = analytic_mea(x[idx], 0, 1e-9)

    # print(np.max(np.abs(mea_x_values - analytic)))

    for t_idx in range(num_tsteps):

        print(t_idx)
        phi = np.load(join(out_folder, "phi_t_vec_{}.npy".format(t_idx)))



    phi_plane = np.zeros((len(x), len(z)))
    for x_idx in range(len(x)):
        for z_idx in range(len(z)):
            phi_plane[x_idx, z_idx] = phi(x[x_idx], 0.0, z[z_idx])


    img1 = ax2.imshow(phi_plane.T, interpolation='nearest', origin='lower', #cmap='bwr',
                      extent=(x[0], x[-1], z[0], z[-1]), vmax=0.6, vmin=-0.6)

    plt.colorbar(img1, ax=ax2)
    l, = ax3.plot(x, mea_x_values,  lw=2, c='g')
    # l2, = ax3.plot(x, analytic,  lw=2, c='k', ls='--')

    plt.savefig(join(fem_fig_folder, 'results_{}_t_idx_{}.png'.format(sim_name, elec)))



if __name__ == '__main__':
    plot_all_results()