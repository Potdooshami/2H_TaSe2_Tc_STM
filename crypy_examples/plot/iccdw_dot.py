from crypy_examples.iccdw_cartoon import (
    atom_draw,
    c_draw,
    ic_draw,
    indc_draw,
    xlim,ylim
)
import matplotlib.pyplot as plt
from useful import fullax



def savepng(draw_fcn,fn):
    plt.figure(figsize=(8,8))
    atom_draw()
        
    draw_fcn()
    indc_draw()
    ax = plt.gca()
    fig = plt.gcf()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fullax(ax)
    fig.savefig("assets/"+fn+".png",dpi=100,bbox_inches='tight',pad_inches=0)

savepng(ic_draw,"iccdw_dot")
savepng(c_draw,"ccdw_dot")