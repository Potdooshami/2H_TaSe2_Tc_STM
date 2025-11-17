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
from crypy_examples.atom_network import gen_atom_Se_simple as gen_atom_Se
# region fundamental input

# region fundamental input
def q3_DWN(n_dom):
    """ generate q3 DWN shift vectors
    Args:
        n_dom (int): single domain size factor
    Returns:
        cry (Crystal2D): crystal including CDW domains            
        cry_supsup (Crystal2D): crystal including super-super structure
    """
    pv = cp.PrimitiveVector2D(a1,a2)
    bss = cp.Basis2D(pv)
    bss.add_artist(gen_atom_Ta,(p1),label='Ta')
    
    bss.add_artist(gen_bond,(p1,p2),label = 'bond1')
    bss.add_artist(gen_bond,(p1,p3),label = 'bond2')
    bss.add_artist(gen_bond,(p2,p4),label = 'bond3')
    bss.add_artist(gen_atom_Se,(p2),label='Se')
    lp = cp.LatticePoints2D(pv)
    cry = cp.Crystal2D(bss,lp)

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

    n_supsup = 2*3*n_dom-2 # this is a unitcell muplier of super-super structure
    pv_supsup = pv.get_super_structure(n_supsup,n_supsup)
    bss_supsup = cp.Basis2D(pv_supsup)
    bss_supsup.add_artist(gen_domain,(0,0),label='domain')
    lp_supsup =  cp.LatticePoints2D(pv_supsup)
    cry_supsup = cp.Crystal2D(bss_supsup,lp_supsup)
    return cry, cry_supsup
