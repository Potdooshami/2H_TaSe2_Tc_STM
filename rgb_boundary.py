import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.patheffects as path_effects

def draw_cyclic_arrows(n, colors=None, reverse=None, radius=1):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Set default colors
    if colors is None:
        colors = ['black'] * n
    elif len(colors) != n:
        raise ValueError(f"colors must have length {n}")
    
    # Set arrow directions
    if reverse is None:
        reverse = [False] * n
    elif len(reverse) != n:
        raise ValueError(f"reverse must have length {n}")
    
    # Compute vertices of the n-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    vertices = np.array([[radius*np.cos(angle), radius*np.sin(angle)] for angle in angles])
    
    # Draw arrows between consecutive vertices
    for i in range(n):
        if reverse[i]:
            # Reverse arrow direction
            start = vertices[(i+1) % n]
            end = vertices[i]
        else:
            # Default arrow direction
            start = vertices[i]
            end = vertices[(i+1) % n]
        
        # Vector from start to end
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        ax.arrow(start[0], start[1], dx, dy, 
                 width=0.05, color=colors[i], length_includes_head=True, 
                 overhang=0.3, head_width=0.15, head_length=0.4)
    
    ax.set_xlim(-radius-0.5, radius+0.5)
    ax.set_ylim(-radius-0.5, radius+0.5)
    ax.set_aspect('equal')
    ax.set_title(f'{n}-gon with cyclic arrows', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.show()

def draw_arrow_trajectory(drs, clrs, show_origin=True, index_points=False, index_arrows=False, text_kwargs=None, xy_origin=(0.0, 0.0)):
    ''' 
    Args
        drs(array): list of direction vectors (dx, dy)
        clrs(n length): list of colors for each arrow
        show_origin(bool): mark the origin (xy_origin)
        index_points(bool): label trajectory points 0..n-1 (start points)
        index_arrows(bool): label arrows 1..n at their midpoints
        text_kwargs(dict): passed to ax.text for styling
        xy_origin(tuple): (x,y) coordinates for the starting origin (default (0,0))
    '''
    drs = np.asarray(drs, dtype=float)
    if drs.ndim != 2 or drs.shape[1] != 2:
        raise ValueError("drs must be an array-like of shape (n, 2)")
    
    n = drs.shape[0]
    if len(clrs) != n:
        raise ValueError("clrs must have the same length as drs")
    
    if text_kwargs is None:
        text_kwargs = dict(color='black', fontsize=10, va='center', ha='center', backgroundcolor=(0, 0, 0, 0), alpha=1, clip_on=True)
    
    fig = plt.gcf()
    ax = plt.gca()
    origin = np.asarray(xy_origin, dtype=float)
    if origin.shape != (2,):
        raise ValueError("xy_origin must be a length-2 sequence (x, y)")
    points = np.zeros((n + 1, 2))
    points[0] = origin
    text_objs = {"points": [], "arrows": []}
    offset = 0.08  # small offset for text placement

    for i, (dx, dy) in enumerate(drs):
        start = points[i]
        ax.arrow(
            start[0], start[1], dx, dy,
            width=0.1,
            length_includes_head=True,
            head_width=0.4,
            head_length=0.4,
            color=clrs[i]
        )
        # label start point if requested (0..n-1)
        if index_points:
            end_ = start + np.array((dx, dy))
            txt = ax.text(end_[0] - offset, end_[1] - offset, str(i+1), **text_kwargs)
            text_objs["points"].append(txt)
        # label arrow (1..n) at midpoint if requested
        if index_arrows:
            mid = start + 0.4 * np.array((dx, dy))
            txt = ax.text(mid[0], mid[1], str(i + 1), **text_kwargs)
            txt.set_path_effects([
                path_effects.withStroke(linewidth=3, foreground='white')
            ])
            text_objs["arrows"].append(txt)
        points[i + 1] = start + np.array((dx, dy))
    
    # mark origin if requested
    if show_origin:
        origin_marker, = ax.plot(origin[0], origin[1], marker='o', color='k', markersize=5)
    
    margin = 0.5
    xmin, xmax = points[:, 0].min() - margin, points[:, 0].max() + margin
    ymin, ymax = points[:, 1].min() - margin, points[:, 1].max() + margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return fig, ax, text_objs

def gen_tex_boundary_words(word, is_color=False):
    '''for given integer list, generate LaTeX code for boundary words.
    1->r, 2->g, 3->b, -1->r^{-1}, -2->g^{-1}, -3->b^{-1}
    Args:
        word (list of int): A list of integers representing the boundary word.\pm(1,2,3)
        is_color (bool): If True, wrap r/g/b in \textcolor{...}{...} for LaTeX coloring.
    Returns:
        str: A string containing the LaTeX code for the boundary word.
    '''
    mapping = {
        1: r'r',
        2: r'g',
        3: r'b',
        -1: r'r^{-1}',
        -2: r'g^{-1}',
        -3: r'b^{-1}',
    }
    color_map = {
        1: 'red',
        2: 'green',
        3: 'blue'
    }

    def fmt(x):
        key = int(x)
        base = mapping[key]
        if is_color:
            c = color_map[abs(key)]
            return rf'\textcolor{{{c}}}{{{base}}}'
        return base

    try:
        return ' '.join(fmt(x) for x in word)
    except (TypeError, ValueError, KeyError):
        raise ValueError("word must be an iterable of integers in [-3, -2, -1, 1, 2, 3]")

class RgbBoundary():
    ALLOWED_VALUES = [1,2,3,-1,-2,-3]
    COLOR = ["red","green","blue"]
    THETAS = np.array([3,1,-1])*np.pi/3
    def __init__(self,word):
        self._word = np.array(word)
        if not self._validate():
            raise ValueError("Invalid word for rgbBoundary.")
        print(self.map_to_one_hot_sign().sum(axis=0))
        if self.is_z3z3_identity():
            print("The word represents the identity element in Z3 x Z3.")
        if self.is_vortex_decomposable():
            print("The word is vortex decomposable.")

    def _validate(self):
        if self._word.size == 0:
            return False
        return np.isin(self._word,self.ALLOWED_VALUES).all()
    def is_vortex_decomposable(self):
        exponent = self.map_to_one_hot_sign().sum(axis=0)
        return (exponent == 0).all()
    def is_z3z3_identity(self):
        exponent = self.map_to_one_hot_sign().sum(axis=0)
        exponent_mod_3 = exponent % 3
        return (exponent_mod_3 == 0).all()
    def map_to_one_hot_sign(self):
        arr = self._word
        n = arr.size

        result = np.zeros((n, 3), dtype=int)

        if n == 0:
            return result
            
        col_indices = np.abs(arr) - 1
        signs = np.sign(arr)
        result[np.arange(n), col_indices] = signs

        return result
    def draw_arrow_trajectory(self,**kwargs):
        drs = []
        for val in self._word:
            tht = self.THETAS[abs(val)-1]
            sign = 1 if val > 0 else -1
            dr = np.array([np.cos(tht), np.sin(tht)]) * sign
            drs.append(dr)
        clrs = [self.COLOR[(abs(val)-1) % 3] for val in self._word]
        fig, ax, text_objs = draw_arrow_trajectory(drs, clrs, **kwargs)
    def draw_cyclic_arrows(self):
        return draw_cyclic_arrows(len(self._word), [self.COLOR[(abs(val)-1) % 3] for val in self._word], [val < 0 for val in self._word], 1)
    def gen_boundary_word_tex(self, is_color=False):
        return gen_tex_boundary_words(self._word, is_color=is_color)

if __name__ == '__main__':
    # Test drawing functions
    RgbBoundary([1,-2,3,-1,2,-3]).draw_arrow_trajectory()
    RgbBoundary([1,2,3]).draw_arrow_trajectory()
    RgbBoundary([-1,-2,-3]).draw_arrow_trajectory()
    RgbBoundary([1,1,1]).draw_arrow_trajectory()
    RgbBoundary([1,2,2]).draw_arrow_trajectory()
    ax = plt.gca()
    ax.set_xlim(-4,4)
    ax.set_ylim(-4,4)
    plt.show()
