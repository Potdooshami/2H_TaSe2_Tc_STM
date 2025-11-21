import pickle
from matplotlib import pyplot as plt
from useful import auto199
from imagingPhase import visPhase as vp
import pandas as pd
import imagingPhase.get_phimap as gpm
import numpy as np
with open('dataCache/pa.pkl','rb') as f:
    pa = pickle.load(f)
with open('dataCache/hddf.pkl','rb') as f:
    df = pickle.load(f)
phiPrinters = []    
for idt in range(len(df)):
    print(idt)
    phiPrinters.append(vp.phiPrinter(pa[0][idt]))

class imgen:
    phiPrinters = phiPrinters
    df = df
    @staticmethod
    def topo(idt):
        img = imgen.df['arr_clns'][idt]
        plt.imshow(img,cmap='afmhot')
        auto199()
    @staticmethod
    def domain(idt):
        Info = imgen.phiPrinters[idt].Info
        phase = imgen.phiPrinters[idt].phase
        vp.DomainColoring(Info).show()
        
    @staticmethod
    def dw(idt):
        Info = imgen.phiPrinters[idt].Info
        phase = imgen.phiPrinters[idt].phase
        vp.DWallColoring(Info,phase).show()
    @staticmethod
    def vertex(idt):
        Info = imgen.phiPrinters[idt].Info
        vp.DVertexColoring(Info).show()
    @staticmethod
    def kmap(idt,ik,iflip):
        ua = np.array([-1, 1])*np.pi
        dfphase = pd.DataFrame({
            'arrfcn':[lambda x:x,gpm.wrap_phase,lambda x: gpm.wrap_phase(3*x)/3],
            'cmap':['jet','twilight_shifted','RdBu'],
            'clim':[ua*3,ua*(1),ua*(1/3)]
        })
        arrfcn = dfphase.iloc[iflip]['arrfcn']
        cmap = dfphase.iloc[iflip]['cmap']
        clim = dfphase.iloc[iflip]['clim']
        arr = imgen.phiPrinters[idt].phase[ik]
        plt.imshow(arrfcn(arr),cmap=cmap)
        auto199()


if __name__ == '__main__':
    for idt in range(len(df)):
        plt.figure()    
        imgen.topo(idt)

        plt.figure()
        imgen.dw(idt)

        plt.figure()
        imgen.vertex(idt)
    # imgen.topo(0)
    plt.show()