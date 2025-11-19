import crypy
import matplotlib.pyplot as plt 
import numpy as np
from crypy_examples.chiral_interlock import lattice_points_in_hex
a1 = [1,0]
a2 = [-0.5,3**0.5/2]
xlim = np.array((-60,60))
ylim = np.array((-60,60))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color = colors[0]
cdw_scatter_kwargs = {'facecolors':'none','edgecolors':color,'s':100}

dlt = .1
n_c = 3


n_ic = n_c + dlt
n_sup = n_ic/dlt

primitve_vector_atom = crypy.PrimitiveVector2D(a1,a2)
primitve_vector_c = primitve_vector_atom.get_super_structure(n_c,n_c)
primitve_vector_ic = primitve_vector_atom.get_super_structure(n_ic,n_ic)
primitve_vector_sup = primitve_vector_atom.get_super_structure(n_sup,n_sup)
lattice_atom = primitve_vector_atom.to_LatticePoints2D()
lattice_ic = primitve_vector_ic.to_LatticePoints2D()
lattice_c = primitve_vector_c.to_LatticePoints2D()
lattice_sup = primitve_vector_sup.to_LatticePoints2D()


lattice_atom.generate_points_by_xylim(xlim,ylim)
lattice_ic.generate_points_by_xylim(xlim,ylim)
lattice_sup.generate_points_by_xylim(xlim*1.2,ylim*1.2)

points_in_hex = lattice_points_in_hex(a1, a2,n_c*3/2+0.01)
lattice_c.generate_points_by_manual(points_in_hex)
gen_dom = lambda x,y:lattice_c.plot_scatter(x,y,**cdw_scatter_kwargs)

basis_sup = primitve_vector_sup.to_Basis2D()
basis_sup.add_artist(gen_dom,(0,0),label='domain')
crystal_sup = crypy.Crystal2D(basis_sup,lattice_sup)











atom_draw = lambda: lattice_atom.plot_scatter(s=2,facecolors=color)
c_draw = lambda: crystal_sup.plot_crystal()
ic_draw = lambda: lattice_ic.plot_scatter(**cdw_scatter_kwargs)
if __name__ == '__main__':
    plt.figure(figsize=(10,10))
    ax = plt.gca()
    # fig,ax = lattice_ic.plot_scatter(facecolors='none',
    #                          edgecolors='red',
    #                          s=100)
    atom_draw()
    # c_draw()
    ic_draw()
    ax = plt.gca()
    limcut = 5*np.array([-1,1])
    ax.set_xlim(xlim-limcut)
    ax.set_ylim(ylim-limcut)
    plt.show()

