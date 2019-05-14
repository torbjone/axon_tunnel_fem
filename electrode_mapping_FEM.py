import os
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import dolfin as df

eps = 1e-9

dx = 200.0
dy = 5.0
dz = 5.0

x0 = -dx / 2
y0 = -dy / 2
z0 = 0.0

x1 = x0 + dx
y1 = y0 + dy
z1 = z0 + dz

nx = 200
ny = 20
nz = 20

sigma = 0.3  # Extracellular conductivity (S/m)


out_folder = 'results'
sim_name = "tunnel_test"
fem_fig_folder = "fem_figs"
[os.makedirs(f, exist_ok=True) for f in [out_folder, fem_fig_folder]]
#
# def analytic_mea(x, y, z):
#     phi = 0
#     for idx in range(len(magnitudes)):
#         r = np.sqrt((x - charge_pos[idx, 0])**2 +
#                     (y - charge_pos[idx, 1])**2 +
#                     (z - charge_pos[idx, 2])**2)
#         phi += magnitudes[idx] / (2 * sigma * np.pi * r)
#     return phi


def plot_FEM_results(phi, elec):
    """ Plot the set-up and potential at xz-plane,
    as well as the potential in the center along the z-axis (depth)
    """

    x = np.linspace(x0, x1, 201)
    z = np.linspace(0, 5, 100)

    mea_x_values = np.zeros(len(x))
    # analytic = np.zeros(len(x))
    for idx in range(len(x)):
        mea_x_values[idx] = phi(x[idx], 0, eps)
        # analytic[idx] = analytic_mea(x[idx], 0, 1e-9)

    # print(np.max(np.abs(mea_x_values - analytic)))

    phi_plane = np.zeros((len(x), len(z)))
    for x_idx in range(len(x)):
        for z_idx in range(len(z)):
            phi_plane[x_idx, z_idx] = phi(x[x_idx], 0.0, z[z_idx])
    plt.close("all")
    fig = plt.figure(figsize=[18, 9])
    # fig.suptitle('Charge positions:\n%s' % (str(charge_pos)))
    fig.subplots_adjust(wspace=0.5, bottom=0.25, top=0.85)
    # ax1 = fig.add_subplot(131, aspect=1, ylabel='z [mm]', xlabel='x [mm]', title='Set up')
    ax2 = fig.add_subplot(211, aspect=1, ylabel='z [mm]', xlabel='x [mm]',
                          title='Field cross section')
    ax3 = fig.add_subplot(212, xlabel='x [mm]', ylabel='Field strength')

    img1 = ax2.imshow(phi_plane.T, interpolation='nearest', origin='lower', #cmap='bwr',
                      extent=(x[0], x[-1], z[0], z[-1]))

    plt.colorbar(img1)
    l, = ax3.plot(x, mea_x_values,  lw=2, c='g')
    # l2, = ax3.plot(x, analytic,  lw=2, c='k', ls='--')

    plt.savefig(join(fem_fig_folder, 'results_{}_elec_{}.png'.format(sim_name, elec)))

def refine_mesh(mesh):

    for r in [2.5]:#[20, 15, 10, 8]:
        print("Refining ...")
        cell_markers = df.MeshFunction("bool", mesh, dim=mesh.topology().dim()-1)
        cell_markers.set_all(False)
        for cell in df.cells(mesh):
            # p = np.sum(np.array(cell.midpoint()[:])**2)
            if np.abs(cell.midpoint()[2]) < r:
                cell_markers[cell] = True
        mesh = df.refine(mesh, cell_markers)

        print(mesh.num_cells())
    mesh.smooth()
    return mesh

# Create classes for defining parts of the boundaries and the interior
# of the domain
class LeftTunnel(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], x0)

class RightTunnel(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], x1)


electrodes = np.arange(-50, 51)

# Initialize sub-domain instances
left = LeftTunnel()
right = RightTunnel()

# Define mesh
mesh = df.BoxMesh(df.Point(x0, y0, z0), df.Point(x1, y1, z1), nx, ny, nz)

print(mesh.num_cells())
# mesh = refine_mesh(mesh)

mesh_coordinates = mesh.coordinates()

np.save(join(out_folder, "mesh_coordinates.npy"), mesh_coordinates)
np.save(join(out_folder, "electrodes.npy"), electrodes)

source_pos = np.load(join(out_folder, "source_pos.npy"))
num_sources = source_pos.shape[0]

# closest_idxs = np.ones(num_sources, dtype=int)
# pos_errors = np.zeros(num_sources)
# for s_idx, s_pos in enumerate(source_pos):
#     closest_idxs[s_idx] = np.argmin(np.sum((mesh_coordinates - s_pos)**2, axis=1))
#     pos_errors[s_idx] = np.sqrt(np.sum((mesh_coordinates[closest_idxs[s_idx]] - s_pos)**2))
#     print(closest_idxs[s_idx], mesh_coordinates[closest_idxs[s_idx]], s_pos, pos_errors[s_idx])


# Initialize mesh function for interior domains
domains = df.MeshFunction("size_t", mesh, mesh.topology().dim())
domains.set_all(0)

# Initialize mesh function for boundary domains
boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
left.mark(boundaries, 1)
# top.mark(boundaries, 2)
right.mark(boundaries, 1)


V = df.FunctionSpace(mesh, "CG", 2)
ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
dx = df.Measure("dx", domain=mesh, subdomain_data=domains)

v = df.TestFunction(V)
u = df.TrialFunction(V)
a = df.inner(sigma * df.grad(u), df.grad(v)) * dx
# Define function space and basis functions

# This corresponds to Neumann boundary conditions zero, i.e. all outer boundaries are insulating.
L = df.Constant(0) * v * dx
# Define Dirichlet boundary conditions at top and bottom boundaries
bcs = [df.DirichletBC(V, 0.0, boundaries, 1)]

for elec in range(len(electrodes)):
    print(elec)
    phi = df.Function(V)
    A = df.assemble(a)
    b = df.assemble(L)

    [bc.apply(A, b) for bc in bcs]

    # Adding point source of magnitude 1.0
    point = df.Point(electrodes[elec], 0.0, 0.0)
    delta = df.PointSource(V, point, 1.0)
    delta.apply(b)

    df.solve(A, phi.vector(), b, 'cg', "ilu")
    phi_sources = np.zeros(num_sources)
    for s_idx, s_pos in enumerate(source_pos):
        phi_sources[s_idx] = phi(s_pos[0], s_pos[1], s_pos[2])

    np.save(join(out_folder, "phi_sources_elec_{}.npy".format(elec)), phi_sources)
    np.save(join(out_folder, "phi_elec_{}.npy".format(elec)), phi.vector())

    plot_FEM_results(phi, elec)


