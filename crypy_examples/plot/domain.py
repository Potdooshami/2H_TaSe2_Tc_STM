import matplotlib.pyplot as plt
from gen_img4pub import (
    imgen
)
from useful import (
    fullax,
    savepng,
    CropWindow
)
from useful import CropWindow
import crypy 
import numpy as np
from crypy_examples.atom_network import color_Se

frame_cntr = 1000,1200
cw = CropWindow()
cw.set_by_anchor(frame_cntr,80)

idt = 4
imgen.topo(idt)

a1 = (20,0)
a2 = (0,20)
o = np.array((996,1199))
a = np.array((1017,1215))
b = np.array((1021,1190))
a1 = a - o
a2 = b - o
# plt.figure(figsize=(5,5))
prim = crypy.PrimitiveVector2D(a1=a1,a2=a2,O=o)
prim_atom = prim.get_sub_structure(3,3)
latt = crypy.LatticePoints2D(prim_atom)
latt.generate_points_by_range((0,2),(0,2))
latt.plot_scatter(color=[0,1,0],s=100)
cw.ax_xylims()
fullax()
f = plt.gcf()
f.set_size_inches(5,5)
savepng(f,'domain')
