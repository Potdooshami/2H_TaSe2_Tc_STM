from gen_img4pub import (
    imgen
)
from imagingPhase import visPhase as vp
from matplotlib import pyplot as plt
from useful import (fullax, savepng,CropWindow)
idt = 0
fig = plt.figure(figsize=(5,5))

xlimylim_DVR = ((560,690),(1310,1440))
xlimylim_DVL = ((1860,1990),(260,390))
dw_r = (1142,1496)
dw_g= (1708,202)
dw_b = (1352,173)

cw_vr = CropWindow()
cw_vr.set_xlimylim(*xlimylim_DVR)
cw_vl = CropWindow()
cw_vl.set_xlimylim(*xlimylim_DVL)
print(cw_vl,cw_vr)
cw_v = [cw_vr,cw_vl]

cw_r = CropWindow()
cw_r.set_by_anchor(dw_r,100)
cw_g = CropWindow()
cw_g.set_by_anchor(dw_g,100)
cw_b = CropWindow()
cw_b.set_by_anchor(dw_b,100)

cw_all = [cw_vr,cw_vl,cw_r,cw_g,cw_b]

fig = plt.figure(figsize=(5,5))
imgen.domain(idt)
ax = plt.gca()
fullax()
savepng(fig,"segment9")

fig = plt.figure(figsize=(5,5))
imgen.dw(idt)
ax = plt.gca()
fullax()
savepng(fig,"solLatt_shiftmap_noCrop")
cw_vr.ax_cropbox()
cw_vl.ax_cropbox()
cw_r.ax_cropbox()
cw_g.ax_cropbox()
cw_b.ax_cropbox()
savepng(fig,"solLatt_shiftmap")
#토포그래피 domain 근접 크롭
#토포그래피 dw 크롭3
# 볼텍스 rL크롭 x 토포,phase,vortex

    

nms_obj = ['vor1','vor2','dw_r','dw_g','dw_b']
fncs = [lambda: imgen.topo(idt),
 lambda: imgen.dw(idt),
 lambda: imgen.vertex(idt)]
nms_fcn = ['topo','dw','vertex']
for cw_v_fcs,nm_obj in zip(cw_all,nms_obj):
    for fcn,nm_fcn in zip(fncs,nms_fcn):
        fig = plt.figure(figsize=(5,5))
        fcn()
        ax = plt.gca()
        fullax()
        cw_v_fcs.ax_xylims()    
        savepng(fig,nm_obj+"_"+nm_fcn)







