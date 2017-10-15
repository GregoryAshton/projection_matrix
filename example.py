import numpy as np
import projection_matrix as pmp

# Generate example data
x = np.linspace(0, 1, 50)
y = np.linspace(20, 30, 60)
z = np.linspace(-2, -1, 70)
x0, y0, z0 = 0.5, 22, -1.5
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
sigmax = 0.1
sigmay = 1
sigmaz = 0.2
D = (np.exp(-(X-x0)**2/sigmax**2)
     + 0.5*np.exp(-(X-0.1)**2/sigmax**2)
     + np.exp(-(Y-y0)**2/sigmay**2)
     + np.exp(-(Z-z0)**2/sigmaz**2))

fig, axes = pmp.projection_matrix(
    D, xyz=[x, y, z], labels=['x', 'y', 'z', 'D'],
    projection=pmp.slice_max,
    #projection=np.max
    )
fig.savefig('example')

