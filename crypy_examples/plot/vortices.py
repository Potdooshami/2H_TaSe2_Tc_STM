from matplotlib import pyplot as plt
from crypy_examples.tripleQ_dwn import q3_DWN
import crypy as cp
import numpy as np

dw_cntr_index = (1/2,0)
w_win = 35
h_win = 40
n_dom =6
domain_range =  ((-1,1),(-1,1))
cry, cry_supsup = q3_DWN(n_dom)

pv_supsup = cry_supsup._lattice.primitive_vector
cntr = pv_supsup.cal_xy_from_ij(dw_cntr_index)
cntr =cntr.flatten()
xlim = (cntr[0]-w_win/2,cntr[0]+w_win/2)
ylim = (cntr[1]-h_win/2,cntr[1]+h_win/2)
cry._lattice.generate_points_by_xylim(xlim,ylim)

gen = lambda x,y: cp.Collection.Generator.gen_hexagon(x=x,y=y,
    c = 'none',edgecolor='k',r = n_dom*3,phi=np.pi/2)
cry_supsup._basis.add_artist(gen,(0,0),label='crop hexagon')
cry_supsup._lattice.generate_points_by_range(*domain_range)
cry.plot_crystal()
# breakpoint()
fig,ax = cry_supsup.plot_crystal()

ax.set_xlim(xlim)
ax.set_ylim(ylim) 

plt.show()