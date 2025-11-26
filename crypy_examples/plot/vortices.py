from matplotlib import pyplot as plt
from crypy_examples.tripleQ_dwn import q3_DWN
from crypy_examples.clr_hexagon import gen_chex
import crypy as cp
import numpy as np

dw_cntr_index = (1/2,0) #center of frame
w_win = 35 # w of frame
h_win = 40 # h of frame
n_dom =6 # domain size
domain_range =  ((-1,1),(-1,1)) 
cry, cry_supsup = q3_DWN(n_dom) # generate crystal

pv_supsup = cry_supsup._lattice.primitive_vector
def get_xlimylim(cntr_index):
    cntr = pv_supsup.cal_xy_from_ij(cntr_index)
    cntr =cntr.flatten()
    xlim = (cntr[0]-w_win/2,cntr[0]+w_win/2)
    ylim = (cntr[1]-h_win/2,cntr[1]+h_win/2)
    return (xlim,ylim)
xlim,ylim = get_xlimylim(dw_cntr_index)
cry._lattice.generate_points_by_xylim(xlim,ylim)



# gen = lambda x,y: cp.Collection.Generator.gen_hexagon(x=x,y=y,
#     c = 'none',edgecolor='k',r = n_dom*3,phi=np.pi/2)
gen_dw = lambda x,y: gen_chex(
    x=x,y=y,R=n_dom*3,c_ord=['g','b','r'],thickness=[1,3],
    alpha=0.5)
cry_supsup._basis.add_artist(gen_dw,(0,0),label='crop hexagon')
cry_supsup._lattice.generate_points_by_range(*domain_range)
cry.plot_crystal()
# breakpoint()
fig,ax = cry_supsup.plot_crystal()

ax.set_xlim(xlim)
ax.set_ylim(ylim)

fig.set_size_inches(w_win/5,h_win/5) 
ax.set_position([0,0,1,1])
ax.set_xticks([])
ax.set_yticks([])
fig.savefig("assets/vortices.png",dpi=150,bbox_inches='tight',pad_inches=0)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
# breakpoint()
# plt.show()