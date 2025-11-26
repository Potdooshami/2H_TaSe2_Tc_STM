import numpy as np  
import matplotlib.pyplot as plt
import crypy
from scipy.ndimage import gaussian_filter1d






genPol = crypy.Collection.Generator.gen_regular_polygon 
gentri = lambda x,y,phi: genPol(n=3,x=x,y=y,r=0.1,c='blue',phi=phi)
def get_tht(x,y,sigma=1):
    dx = np.gradient(gaussian_filter1d(x, sigma=sigma))
    dy = np.gradient(gaussian_filter1d(y, sigma=sigma))
    return np.arctan2(dy,dx)
# breakpoint() 
def draw_on_line(xs,ys,thts,gen):
    for x,y,tht in zip(xs,ys,thts):
        gen(x,y,tht)
    

#     return x*np.cos(x)
if __name__ == '__main__':
    x = np.linspace(0,10,100)
    y = x*np.cos(x)
    plt.plot(x,y)
    tht = get_tht(x,y)
    plt.plot(x,tht)
    # dx = gaussian_filter1d(x, sigma=1)
    # dy = gaussian_filter1d(y, sigma=1)
    # plt.plot(x,dx)
    # plt.plot(x,dy)
    gentri(0,0,phi=0)
    plt.axhline(y=0,color='k')
    plt.axis('image')
    draw_on_line(x,y,tht,gentri)
    plt.show()
    
    