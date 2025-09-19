import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from ..useful import *
from . import auto199
def sigmoid(x, A, L, x0, k):
    """
    Sigmoid function with controllable parameters.

    Parameters:
    - x: Independent variable (numpy array or scalar)
    - A: Lower asymptote (step bottom)
    - L: Upper asymptote (step top)
    - x0: Midpoint (step position)
    - k: Steepness/gradient (step gradient)
    """
    return A + (L - A) / (1 + np.exp(-k * (x - x0)))
class stepRecover:
    def __init__(self,arr_steped,y_affected=None,x_lims=None,y_lims=None,y_fit=None):
        self.arr_steped = arr_steped
        if y_affected == None:
            y_affected = arr_steped.shape[0]/2
        if x_lims == None:
            x_lims = (0, arr_steped.shape[0]-1)
        if y_lims == None:
            y_lims = (0, arr_steped.shape[1]-1)
        self.y_affected = y_affected            
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.y_fit = y_fit
        self.x = np.arange(arr_steped.shape[0]) 
    def get_xyroi(self):
        y_lims = self.y_lims
        x_lims = self.x_lims
        self.x_roi = np.arange(*y_lims)
        self.y_roi = self.arr_steped[y_lims[0]:y_lims[1],x_lims[0]:x_lims[1]].mean(axis=1)
        print(self.y_roi.shape)
    def get_p0(self):
        initial_A = np.min(self.y_lims)
        initial_L = np.max(self.y_lims)
        initial_x0 = self.y_affected
        initial_k = 1.0
        self.p0 =  [initial_A, initial_L, initial_x0, initial_k]
    def fit(self):
        print('fitted')
        self.get_xyroi()
        self.get_p0()
        popt, pcov = curve_fit(sigmoid, self.x_roi, self.y_roi, p0=self.p0, maxfev=5000)
        self.p_fit = popt
        self.y_fit =  sigmoid(self.x,*popt)
        self.arr_flat = self.arr_steped - self.y_fit.reshape(-1,1)
    def imshow1d(self):
        self.get_xyroi()
        plt.plot(self.x_roi,self.y_roi,color = 'g')
        plt.axvline(self.y_affected,color = 'r')
        plt.xlabel('y')
        if self.y_fit is None:
            pass
        else:
            plt.plot(self.x,self.y_fit,color = 'k')
    def imshow2d(self,arr):
        plt.imshow(arr,cmap='jet')
        auto199()
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')        
    def imshow(self):
        from matplotlib.patches import Rectangle
        fig,axs = plt.subplots(2,2)
        tnss = np.array([['stepped', '1d_full'], ['flatten', '1d_fcs']])

        for ax, title in zip(axs.flat, tnss.flat):
            ax.set_title(title)

        ax = axs[0,0]
        plt.sca(ax)
        self.imshow2d(self.arr_steped)
        plt.axhline(self.y_affected,color = 'r')
        x_lims =self.x_lims
        y_lims =self.y_lims
        rect = Rectangle(
            (x_lims[0], y_lims[0]), 
            x_lims[1]-x_lims[0], y_lims[1]-y_lims[0], 
            linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

        ax = axs[1,0]
        plt.sca(ax)
        if self.y_fit is None:
            pass
        else:
            self.imshow2d(self.arr_flat)
        
        ax = axs[0,1]
        plt.sca(ax)
        self.imshow1d()
        # plt.xlim(self.)

        ax = axs[1,1]
        plt.sca(ax)
        self.imshow1d()
        plt.xlim(self.y_lims[0],self.y_lims[1])



        fig.tight_layout()
        return (fig,axs)
# if __name__ == "__main__":
#     import imagingPhase.surgery as sgr