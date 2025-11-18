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

if __name__ == '__main__':
    for idt in range(len(df)):
        plt.figure()    
        img =df['arr_clns'][idt]
        plt.imshow(img,cmap='afmhot')
        auto199()

        plt.figure()    
        Info = phiPrinters[idt].Info
        phase = phiPrinters[idt].phase  
        vp.DWallColoring(Info,phase).show()

        plt.figure()
        rgb = vp.DVertexColoring(Info).show()    
    plt.show()