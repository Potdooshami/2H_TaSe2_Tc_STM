import crypy as cp
import numpy as np
from matplotlib import pyplot as plt
from crypy_examples.atom_network import draw_atom, draw_bond
from crypy_examples.chiral_interlock import lattice_points_in_hex


# region Fundamental Parameters
a1 = [1,0] # lattic unitcell vector 1
a2 = [-0.5,3**0.5/2] # lattic unitcell vector 2
p1 = np.array((2,1))/3 # basis point1
p2 = np.array((1,2))/3 # basis point2
p3=np.array((1,-1))/3  # basis point3
p4=-p3 # basis point4

n_dom =18 # single Domain's size factor
n_supsup = 2*3*n_dom-2
domain_range =  ((-1,1),(-1,1))
n__ = 2*n_supsup
latt_range = ((-n__,n__),(-n__,n__))
# endregion

# region class setup for figure
# region LEVEL 1: ATOMIC LATTICE
pv = cp.PrimitiveVector2D(a1,a2)
bss = cp.Basis2D(pv)
gen_atom_Ta = lambda x,y: draw_atom(x, y, radius=0.2, color_hex='#4169E1')
gen_atom_Se = lambda x,y: draw_atom(x, y, radius=0.1, color_hex='#FF6347')
gen_bond = lambda x,y: draw_bond(x,y,r=.05,facecolor=(.7,.7,.7))


bss.add_artist(gen_atom_Ta,(p1),label='Ta')
bss.add_artist(gen_atom_Se,(p2),label='Se')
bss.add_artist(gen_bond,(p1,p2),label = 'bond1')
bss.add_artist(gen_bond,(p1,p3),label = 'bond2')
bss.add_artist(gen_bond,(p2,p4),label = 'bond3')
# xylim = ((-10,10),(-10,10))

lp = cp.LatticePoints2D(pv) 
# lp.generate_points_by_xylim(*xylim)
lp.generate_points_by_range(*latt_range)
cry = cp.Crystal2D(bss,lp)
# endregion

# region LEVEL 2: CDW 
pv_sup = pv.get_super_structure(3,3)
bss_sup = cp.Basis2D(pv_sup)
gen_CDW = lambda xxx,yyy: plt.fill(xxx,yyy,"y",alpha=.3)
# cdw = dict()
cdw_p1 = (2/3,1/3)
cdw_p2 = (-1/3,1/3)
cdw_p3 = (-1/3,-2/3)
cdw_ps = (cdw_p1,cdw_p2,cdw_p3)
bss_sup.add_artist(gen_CDW,cdw_ps,label='CDW')





lp_sup = cp.LatticePoints2D(pv_sup) 
# lp_sup.generate_points_by_xylim(*xylim)


ps = lattice_points_in_hex(a1, a2,n_dom)
lp_sup.generate_points_by_manual(ps)
cry_sup = cp.Crystal2D(bss_sup,lp_sup)
gen_domain = lambda x,y:cry_sup.plot_crystal(x,y)
# endregion

# region LEVEL 3: SUPER-SUPER STRUCTURE
pv_supsup = pv.get_super_structure(n_supsup,n_supsup)
bss_supsup = cp.Basis2D(pv_supsup)
bss_supsup.add_artist(gen_domain,(0,0),label='domain')
lp_supsup =  cp.LatticePoints2D(pv_supsup)
lp_supsup.generate_points_by_range(*domain_range)
cry_supsup = cp.Crystal2D(bss_supsup,lp_supsup)
# endregion
# endregion

# region plot

#cry_sup._basis.plot_basis()
# cry_sup._basis.primitive_vector.plot_all()
#cry_sup._lattice.primitive_vector.plot_all()

# cry_sup._lattice.plot_scatter()

# fig,ax  = cry.plot_crystal()
# ax.set_xlim(-8,8)
# ax.set_ylim(-8,8)

cry_supsup.plot_crystal()
# cry.plot_crystal()
plt.show()


# endregion
