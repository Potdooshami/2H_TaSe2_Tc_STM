import crypy as cp
import numpy as np
from matplotlib import pyplot as plt
import copy
import matplotlib.patches as patches
from matplotlib.path import Path
from shapely.geometry import Polygon, box

from crypy_examples.atom_network import (
    draw_atom, 
    draw_bond,
    color_bond,
    r_Se,
    r_Ta,
    r_bond,
    a1,a2,
    p1,p2,p3,p4        
)
from matplotlib.colors import to_rgb
from crypy_examples.colorspace import CLR_NODE

xylim = ((-10,10),(-10,10))

color_Se_HA = [254,164,0]
color_Ta_HA = [0, 176, 240]
color_Se_vesta = [202,137,33]
color_Ta_vesta = [38,115,148]
hexer =  lambda rgb: '#{:02x}{:02x}{:02x}'.format(*rgb)

pv = cp.PrimitiveVector2D(a1,a2)
bss = cp.Basis2D(pv)

alpha_common = .3
def gen_cation_triangle(x,y):
    return cp.Collection.Generator.gen_regular_polygon(
        3,x=x,y=y,r = 1/np.sqrt(3),alpha=alpha_common,
        c=CLR_NODE[1],phi=-np.pi/6
        )
def gen_anion_triangle(x,y):
    return cp.Collection.Generator.gen_regular_polygon(
        3,x=x,y=y,r = 1/np.sqrt(3),alpha=alpha_common,
        c=CLR_NODE[0],phi=np.pi/6
        )

gen_atom_Ta = lambda x,y: draw_atom(x, y, radius=r_Ta*.6*1.5, color_hex=hexer(color_Ta_HA))
gen_atom_Se = lambda x,y: draw_atom(x, y, radius=r_Se*1.5, color_hex=hexer(color_Se_HA))
gen_bond = lambda x,y: draw_bond(x,y,r=r_bond*.6,facecolor=to_rgb(color_bond))

bss.add_artist(gen_bond,(p1,p2),label = 'bond1')
bss.add_artist(gen_bond,(p1,p3),label = 'bond2')
bss.add_artist(gen_bond,(p2,p4),label = 'bond3')
bss.add_artist(gen_atom_Ta,(p1),label='Ta')
bss.add_artist(gen_atom_Se,(p2),label='Se')

lp = cp.LatticePoints2D(pv) 
lp.generate_points_by_xylim(*xylim)
cry = cp.Crystal2D(bss,lp)

pv_sup = pv.get_super_structure(3,3)
bss_sup = cp.Basis2D(pv_sup)
gen_CDW = lambda xxx,yyy: plt.fill(xxx,yyy,"y",alpha=.3)
cdw_p1 = np.array((2/3,1/3))
cdw_p2 = np.array((-1/3,1/3))
cdw_p3 = np.array((-1/3,-2/3))
cdw_ps = np.array((cdw_p1,cdw_p2,cdw_p3)) + np.array((1/3,0)).reshape(1,2)

bss_sup.add_artist(gen_CDW,cdw_ps,label='CDW')
lp_sup = cp.LatticePoints2D(pv_sup)   
lp_sup.generate_points_by_xylim(*xylim)
cry_sup = cp.Crystal2D(bss_sup,lp_sup)
gen_domain = lambda x,y:cry_sup.plot_crystal(x,y)
gen_wigner = lambda x,y: cp.Collection.Generator.gen_hexagon(
    x=x,y=y,r=3*1/np.sqrt(3),alpha=.3,phi=np.pi/2,facecolor='none',edgecolor='m',lw=1)
bss_wigner = cp.Basis2D(pv_sup)
bss_wigner.add_artist(gen_wigner,(0,0),label='wigner')
cry_sup_wigner = cp.Crystal2D(bss_wigner,lp_sup)

cry._basis.add_artist(gen_cation_triangle,(p1),label='cation_triangle')
cry._basis.add_artist(gen_anion_triangle,(p2),label='anion_triangle')

cry_lattice = copy.deepcopy(cry)
cry_vortex_triangle = copy.deepcopy(cry)
cry_lattice._basis._artist_list = cry_lattice._basis._artist_list[:-2]
cry_vortex_triangle._basis._artist_list = cry_vortex_triangle._basis._artist_list[-2:]

def get_subtracted_patch():
    """
    Creates a square and a regular hexagon using Shapely, performs a subtraction
    (Square - Hexagon), and returns the result as a Matplotlib PathPatch object.
    
    Returns:
        patches.PathPatch: The subtracted patch object with a hole in the center.
    """
    # 1. Create a Shapely geometry for the Square
    square_geom = box(-5, -5, 5, 5)

    # 2. Create a Shapely geometry for the Regular Hexagon
    hex_radius = np.sqrt(3)
    num_sides = 6
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False) + np.pi / 2
    hex_vertices = [(hex_radius * np.cos(a), hex_radius * np.sin(a)) for a in angles]
    hexagon_geom = Polygon(hex_vertices)

    # 3. Perform the subtraction (Square - Hexagon)
    result_geom = square_geom.difference(hexagon_geom)

    # 4. Convert the Shapely result into a Matplotlib PathPatch
    if result_geom.geom_type == 'Polygon':
        ext_vertices = np.array(result_geom.exterior.coords)
        
        # Initialize lists for vertices and path codes
        all_vertices = [ext_vertices]
        all_codes = [[Path.MOVETO] + [Path.LINETO] * (len(ext_vertices) - 2) + [Path.CLOSEPOLY]]
        
        # Process interior holes (the hexagon part)
        for interior in result_geom.interiors:
            int_vertices = np.array(interior.coords)
            all_vertices.append(int_vertices)
            all_codes.append([Path.MOVETO] + [Path.LINETO] * (len(int_vertices) - 2) + [Path.CLOSEPOLY])
        
        # Merge all components into single numpy arrays
        vertices = np.concatenate(all_vertices)
        codes = np.concatenate(all_codes)
        
        # Create the final Matplotlib PathPatch object
        subtracted_path = Path(vertices, codes)
        subtracted_patch = patches.PathPatch(
            subtracted_path,
            linewidth=2,
            edgecolor='magenta',
            facecolor='lightblue',
            alpha=1,
            zorder=1000000,
            label='Square - Hexagon'
        )
        return subtracted_patch
        
    return None

if __name__ == '__main__':
    fig, ax = cry.plot_crystal()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.show()
