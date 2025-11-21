from gen_img4pub import imgen
# breakpoint()
from matplotlib import pyplot as plt
from useful import fullax,savepng
for ik in range(3):
    fig = plt.figure(figsize=(5,5))
    imgen.kmap(2,ik,1)
    fullax()
    savepng(fig,f'iccdw_kmap3_ik{ik}')
    for idt in range(5):
        fig = plt.figure(figsize=(5,5))
        imgen.kmap(idt,ik,2)
        fullax()
        savepng(fig, f'iccdw_kmap_ik{ik}_idt{idt}')
