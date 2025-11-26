import crypy
import matplotlib.pyplot as plt 
import numpy as np
from crypy_examples.chiral_interlock import lattice_points_in_hex
# re
a1 = [1,0]
a2 = [-0.5,3**0.5/2]

xlim = np.array((-60,60))
ylim = np.array((-60,60))

from matplotlib.colors import to_rgba
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color = colors[0]
# cdw_scatter_kwargs = {'facecolors':to_rgba(color,0.2),'edgecolors':color,'s':100}
cdw_scatter_kwargs = {'facecolors':to_rgba(color,0.2),'edgecolors':'none','s':100}
# breakpoint()
indicator_kwargs = {'facecolors':to_rgba('red',0.5),'edgecolors':'red','s':100}

cdw_before_kwargs = {'facecolors':to_rgba([0,1,1],.3),'edgecolors':[1,0,0,0],'s':100}
cdw_after_kwargs = {'facecolors':to_rgba('blue',0),'edgecolors':[0,1,1],'s':100}
# cdw_before_kwargs = {'edgecolors':'green','s':100}
# cdw_after_kwargs = {'edgecolors':'blue','s':100}


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

lattice_clong = primitve_vector_c.to_LatticePoints2D()
lattice_clong.generate_points_by_xylim(xlim,ylim)


atom_draw = lambda: lattice_atom.plot_scatter(s=2,facecolors=color)
c_draw = lambda: crystal_sup.plot_crystal()
clong_draw = lambda: lattice_clong.plot_scatter(**cdw_scatter_kwargs)
ic_draw = lambda: lattice_ic.plot_scatter(**cdw_scatter_kwargs)
indc_draw = lambda: lattice_sup.plot_scatter(**indicator_kwargs)

import copy
def create_crystal_sup(**kwargs):
    crystal_sup_new = copy.deepcopy(crystal_sup)
    crystal_sup_new._basis._artist_list = []
    gen_dom = lambda x,y:lattice_c.plot_scatter(x,y,**kwargs)
    crystal_sup_new._basis.add_artist(gen_dom,(0,0),label='domain')
    return crystal_sup_new

def cong2c_draw():
    lattice_clong.plot_scatter(**cdw_before_kwargs)
    foo =create_crystal_sup(**cdw_after_kwargs)
    foo.plot_crystal()

    
def c2ic_draw():
    foo =create_crystal_sup(**cdw_before_kwargs)
    foo.plot_crystal()
    lattice_ic.plot_scatter(**cdw_after_kwargs)
    




if __name__ == '__main__':
    plt.figure(figsize=(10,10))
    ax = plt.gca()
    # fig,ax = lattice_ic.plot_scatter(facecolors='none',
    #                          edgecolors='red',
    #                          s=100)
    atom_draw()
    # plt.figure(figsize=(10,10))
    # c_draw()
    ic_draw()
    indc_draw()
    # clong_draw()
    # crystal_sup_new = create_crystal_sup(**cdw_before_kwargs)
    # crystal_sup_new.plot_crystal()
    # cong2c_draw()
    # c2ic_draw()
    # ax = plt.gca()
    limcut = 5*np.array([-1,1])
    ax.set_xlim(xlim-limcut)
    ax.set_ylim(ylim-limcut)
    plt.show()

