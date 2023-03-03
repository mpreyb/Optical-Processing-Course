"""
    File name: stepfib_modes.py
    Author: Maria Paula Rey
    Email : mpreyb@eafit.edu.co
    Date last modified: 19/11/2021
    Python Version: 3.8
"""

import numpy as np
import matplotlib.pyplot as plt
import meshio 
from scipy.linalg import eigh
from scipy import interpolate

n_core = 1.466
n_clad = 1.441

#----------------------------------------------------------------------------------------------

# assem(pts[:, :2], core)
def assem(coords, elems, rindx):
    """
    Assembles stiffness and mass matrix
    
    Parameters
    ----------
    coords : ndarray, float
        Node coordinates
    elems : ndarray, int
        Element connectivity
    rindx : 
        Refractive index of material
    Returns
    -------
    stiff : ndarray, float
        Stiffness matrix.
    mass : ndarray, float
        Mass matrix
    """
    ncoords = coords.shape[0]
    stiff = np.zeros((ncoords, ncoords))
    mass = np.zeros((ncoords, ncoords))
    for el_cont, elem in enumerate(elems):
        rindx_prom = np.sum(rindx[elem])/3
        stiff_loc, mass_loc = local_mat(coords[elem], rindx_prom)
        for row in range(3):
            for col in range(3):
                row_glob, col_glob = elem[row], elem[col]
                stiff[row_glob, col_glob] += stiff_loc[row, col]
                mass[row_glob, col_glob] += mass_loc[row, col]
    return stiff, mass


#---------------------------------------------------------------------------------------

def local_mat(coords, rindx):
    """
    Calculates local stiffness matrix
        
    Parameters
    ----------
    coords : ndarray, float
        Element node coordinates
    rindx : 
        Refractive index of material
    
    Returns
    -------
    stiff : ndarray, float
        Local stiffness matrix
    mass : float
        Local mass matrix
    """
    dNdr = np.array([
            [-1, 1, 0],
            [-1, 0, 1]])
    jaco = dNdr @ coords
    det = np.linalg.det(jaco)
    jaco_inv = np.linalg.inv(jaco)
    dNdx = jaco_inv @ dNdr
    stiff = 0.5 * det * (dNdx.T @ dNdx) *rindx
    mass = det/24 * np.array([
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0]])
    return stiff, mass

#---------------------------------------------------------------------------------------

def pts_restringidos(coords, malla, lineas_rest):
    """
    Identifies restricted and free nodes for the mesh.
        
    Parameters
    ----------
    coords : ndarray, float
        Node element coordinates
    malla : Meshio mesh
        Mesh (object)
    lineas_rest : list
        List with numbers for the restricted lines
    
    Returns
    -------
    pts_rest : list
        Restricted points list
    pts_libres : list
        Free points list
    """
    lineas = [malla.cells[k].data for k in lineas_rest]
    pts_rest = []
    for linea in lineas:
        pts_rest += set(linea.flatten())
    pts_libres = list(set(range(coords.shape[0])) - set(pts_rest))
    return pts_rest, pts_libres

#---------------------------------------------------------------------------------------

# Reading and storing mesh
malla = meshio.read("circle_refined.msh")
pts = malla.points                          # Mesh points
x, y = pts[:, 0:2].T

# Storing cladding elements
core_1 = malla.cells[9].data                      
core_2 = malla.cells[10].data   
core_3 = malla.cells[11].data   
core_4 = malla.cells[12].data

# Storing core elements 
clad_1 = malla.cells[13].data                     
clad_2 = malla.cells[14].data 
clad_3 = malla.cells[15].data 
clad_4 = malla.cells[16].data 

# Graphing mesh
# plt.figure()
# plt.triplot(x, y, core_1, linewidth=0.2)
# plt.triplot(x, y, core_2, linewidth=0.2)
# plt.triplot(x, y, core_3, linewidth=0.2)
# plt.triplot(x, y, core_4, linewidth=0.2)

# plt.triplot(x, y, clad_1, linewidth=0.2)
# plt.triplot(x, y, clad_2, linewidth=0.2)
# plt.triplot(x, y, clad_3, linewidth=0.2)
# plt.triplot(x, y, clad_4, linewidth=0.2)

# plt.axis("image")
# plt.show()

#---------------------------------------------------------------------------------------

# Combining numpy arrays for core and cladding
tri = np.concatenate(([malla.cells[i].data for i in range(9, 17)]), axis=0)

r = np.sqrt(x**2 + y**2)     # Radius value
r_core = 12                  # Radius of core-cladding interface. Modified according to msh file
source = np.piecewise(r, [r <= r_core, r > r_core], [n_core, n_clad])

 # Assembling stiffness and mass matrixes
stiff, mass = assem(pts[:, :2], tri, source)  

 # Defining free and restricted points
pts_rest, pts_libres = pts_restringidos(pts, malla, [5, 6, 7, 8])

nvals = 36         # Desired numbers of eigenvalues
vals, vecs = eigh(stiff[np.ix_(pts_libres, pts_libres)],
                  mass[np.ix_(pts_libres, pts_libres)],
                  eigvals=(0, nvals - 1))     #solving

np.round(vals, 4)

# Plotting our results
plt.figure(figsize=(6, 4))
plt.plot(range(1, nvals + 1), vals, "o", color="green")
plt.xlabel("Eigenvalue", fontsize=16)
plt.ylabel(r"$\lambda_{mn}^2$", fontsize=16)
plt.show()

sol_c = np.zeros(pts.shape[0])
sol_c[pts_libres] = vecs[:, 10]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_trisurf(x, y, sol_c, triangles=tri, cmap="RdYlBu")
ax.set_xlabel(r"$x$", fontsize=16)
ax.set_ylabel(r"$y$", fontsize=16)
ax.set_zlabel(r"$u$", fontsize=16)
plt.show()

X, Y = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
X, Y = np.meshgrid(X, Y)

rbf = interpolate.Rbf(x, y, sol_c, function='linear')
Z = rbf(X, Y)

plt.figure()
plt.imshow(Z, vmin=sol_c.min(), vmax=sol_c.max(), origin='lower',
           extent=[x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.show()