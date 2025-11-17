import crypy as cp
import numpy as np
from matplotlib import pyplot as plt
from crypy_examples.chiral_interlock import lattice_points_in_hex
from crypy_examples.atom_network import (
    a1,a2,
    p1,p2,p3,p4,
    gen_atom_Se,
    gen_bond        
)
from crypy_examples.atom_network import gen_atom_Ta_v2 as gen_atom_Ta

# region Fundamental Parameters
ind_dw = 0 # from0 to 3
n_dom =18 # single Domain's size factor; 3 times of CDW unitcell
n_supsup = 2*3*n_dom-2 # this is a unitcell muplier of super-super structure
domain_range =  ((0,1),(0,1))
# region crop window
l_sq = 30
dw_cntr_indices = ((1/2,0),(0,1/2),(1/2,1/2))

# endregion
# endregion

# region class setup for figure
# region LEVEL 1: ATOMIC LATTICE
pv = cp.PrimitiveVector2D(a1,a2)
bss = cp.Basis2D(pv)


bss.add_artist(gen_atom_Ta,(p1),label='Ta')
bss.add_artist(gen_atom_Se,(p2),label='Se')
bss.add_artist(gen_bond,(p1,p2),label = 'bond1')
bss.add_artist(gen_bond,(p1,p3),label = 'bond2')
bss.add_artist(gen_bond,(p2,p4),label = 'bond3')
# xylim = ((-10,10),(-10,10))

# endregion

# region LEVEL 2: CDW 
pv_sup = pv.get_super_structure(3,3)
bss_sup = cp.Basis2D(pv_sup)
gen_CDW = lambda xxx,yyy: plt.fill(xxx,yyy,"y",alpha=.3)
cdw_p1 = (2/3,1/3)
cdw_p2 = (-1/3,1/3)
cdw_p3 = (-1/3,-2/3)
cdw_ps = (cdw_p1,cdw_p2,cdw_p3)
bss_sup.add_artist(gen_CDW,cdw_ps,label='CDW')





lp_sup = cp.LatticePoints2D(pv_sup) 
ps = lattice_points_in_hex(a1, a2,n_dom)
lp_sup.generate_points_by_manual(ps)
cry_sup = cp.Crystal2D(bss_sup,lp_sup)
gen_domain = lambda x,y:cry_sup.plot_crystal(x,y)
# endregion

# region LEVEL 3: SUPER-SUPER STRUCTURE
pv_supsup = pv.get_super_structure(n_supsup,n_supsup)
# region crop window
dw_cntr_index = dw_cntr_indices[ind_dw]
cntr = pv_supsup.cal_xy_from_ij(dw_cntr_index)
cntr =cntr.flatten()
xlim = (cntr[0]-l_sq/2,cntr[0]+l_sq/2)
ylim = (cntr[1]-l_sq/2,cntr[1]+l_sq/2)
# endregion
lp = cp.LatticePoints2D(pv) 
lp.generate_points_by_xylim(xlim,ylim)
cry = cp.Crystal2D(bss,lp)


bss_supsup = cp.Basis2D(pv_supsup)
bss_supsup.add_artist(gen_domain,(0,0),label='domain')
lp_supsup =  cp.LatticePoints2D(pv_supsup)
lp_supsup.generate_points_by_range(*domain_range)
cry_supsup = cp.Crystal2D(bss_supsup,lp_supsup)
# endregion
# endregion

# region plot
plt.figure(figsize=(6,6))
# cry.plot_crystal()
fig,ax  = cry_supsup.plot_crystal()
ax.set_xticks([])
ax.set_yticks([])
ax.set_axis_off()
ax.set_position([0,0,1,1])




cry.plot_crystal()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig("dw3.png",dpi=300,bbox_inches='tight',pad_inches=0)
# plt.show()


# endregion
