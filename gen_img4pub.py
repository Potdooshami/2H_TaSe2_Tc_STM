import pickle
import matplotlib.pyplot as plt
from useful import auto199
from imagingPhase import visPhase as vp
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
    def dw(idt):
        Info = imgen.phiPrinters[idt].Info
        phase = imgen.phiPrinters[idt].phase
        vp.DWallColoring(Info,phase).show()
    @staticmethod
    def vertex(idt):
        Info = imgen.phiPrinters[idt].Info
        vp.DVertexColoring(Info).show()



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