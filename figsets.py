from matplotlib import pyplot as plt
import numpy as np



class PanelChild:
    """Individual panel within a publication figure"""
    
    def __init__(self, parent_fig, lbwh, label=None,comment='...'):
        """
        Parameters
        ----------
        parent_fig : pubfig
            Parent figure object
        lbwh : array-like
            [left, bottom, width, height] in mm
        label : str, optional
            Label for the panel
        """
        self.parent = parent_fig
        self.lbwh = np.asarray(lbwh, dtype=float)
        self.label = label if label is not None else 'unknown'
        self.comment = comment        
        # Create axes with normalized coordinates
        lbwh_axu = self.lbwh * np.array([1, 1/parent_fig.height_u, 1, 1/parent_fig.height_u])
        self.ax = parent_fig.fig.add_axes(lbwh_axu)
    
    @property
    def left(self):
        return self.lbwh[0]
    
    @property
    def bottom(self):
        return self.lbwh[1]
    
    @property
    def width(self):
        return self.lbwh[2]
    
    @property
    def height(self):
        return self.lbwh[3]
    
    def resize(self, width=None, height=None, anchor='bottom_left'):
        """
        Resize the panel while keeping an anchor point fixed
        
        Parameters
        ----------
        width : float, optional
            New width in mm
        height : float, optional
            New height in mm
        anchor : str or tuple, optional
            Anchor point to keep fixed during resize.
            Default is 'bottom_left' (maintains left, bottom position)
            Options: 'center', 'top', 'bottom', 'left', 'right',
                    'top_left', 'top_right', 'bottom_left', 'bottom_right'
                    or (x_frac, y_frac) where 0 <= frac <= 1
        """
        # Get anchor point before resize
        anchor_point = self.get_point(anchor)
        
        # Apply new dimensions
        if width is not None:
            self.lbwh[2] = width
        if height is not None:
            self.lbwh[3] = height
        
        # Calculate how much the anchor point would move
        new_anchor_point = self.get_point(anchor)
        
        # Adjust position to keep anchor point fixed
        self.lbwh[0] += (anchor_point[0] - new_anchor_point[0])
        self.lbwh[1] += (anchor_point[1] - new_anchor_point[1])
        
        self._update_axes()
    
    def reduce(self, w_reduce=0, h_reduce=0, anchor='bottom_left'):
        """
        Reduce the panel size while keeping an anchor point fixed
        
        Parameters
        ----------
        w_reduce : float, optional
            Amount to reduce width in mm (default: 0)
        h_reduce : float, optional
            Amount to reduce height in mm (default: 0)
        anchor : str or tuple, optional
            Anchor point to keep fixed during reduction.
            Default is 'bottom_left' (maintains left, bottom position)
            Options: 'center', 'top', 'bottom', 'left', 'right',
                    'top_left', 'top_right', 'bottom_left', 'bottom_right'
                    or (x_frac, y_frac) where 0 <= frac <= 1
        """
        new_width = self.width - w_reduce
        new_height = self.height - h_reduce
        self.resize(width=new_width, height=new_height, anchor=anchor)
    
    def translate(self, dx=0, dy=0):
        """Translate the panel by dx, dy in mm"""
        self.lbwh[0] += dx
        self.lbwh[1] += dy
        self._update_axes()
    
    def set_position(self, left=None, bottom=None):
        """Set absolute position"""
        if left is not None:
            self.lbwh[0] = left
        if bottom is not None:
            self.lbwh[1] = bottom
        self._update_axes()
    
    def get_point(self, anchor='center'):
        """
        Get a point on the panel boundary
        
        Parameters
        ----------
        anchor : str or tuple
            'center', 'top', 'bottom', 'left', 'right', 
            'top_left', 'top_right', 'bottom_left', 'bottom_right'
            or (x_frac, y_frac) where 0 <= frac <= 1
        
        Returns
        -------
        point : ndarray
            [x, y] coordinates in mm
        """
        if isinstance(anchor, tuple):
            x_frac, y_frac = anchor
            return np.array([
                self.left + self.width * x_frac,
                self.bottom + self.height * y_frac
            ])
        
        anchor_map = {
            'center': (0.5, 0.5),
            'top': (0.5, 1.0),
            'bottom': (0.5, 0.0),
            'left': (0.0, 0.5),
            'right': (1.0, 0.5),
            'top_left': (0.0, 1.0),
            'top_right': (1.0, 1.0),
            'bottom_left': (0.0, 0.0),
            'bottom_right': (1.0, 0.0),
        }
        
        if anchor not in anchor_map:
            raise ValueError(f"Unknown anchor: {anchor}")
        
        return self.get_point(anchor_map[anchor])
    
    def _update_axes(self):
        """Update axes position after transformation"""
        lbwh_axu = self.lbwh * np.array([1, 1/self.parent.height_u, 1, 1/self.parent.height_u])
        self.ax.set_position(lbwh_axu)
    
    def plot_layout(self):
        """Draw layout helper (cross and label)"""
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.plot([0, 1], [0, 1], 'k', linewidth=0.1)
        self.ax.plot([1, 0], [0, 1], 'k', linewidth=0.1)
        self.ax.text(0, 1, self.label, transform=self.ax.transAxes, 
                    ha='left', va='top', fontsize=12)
        self.ax.text(.5, .5, self.comment, transform=self.ax.transAxes, 
                    ha='center', va='center', fontsize=6)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)


class pubfig:
    MM_PER_INCH = 25.4
    WIDTH_2COL = 178
    WIDTH_1COL = 86
    
    def __init__(self, width, height_u):        
        if width == '1col':
            self.width = pubfig.WIDTH_1COL
        elif width == '2col':
            self.width = pubfig.WIDTH_2COL
        else:
            self.width = width
        self.height_u = height_u            
        self.fig = plt.figure(figsize=(pubfig.mm_to_inch(self.width), 
                                       pubfig.mm_to_inch(self.height)))
        self.children = []
    
    @staticmethod
    def mm_to_inch(mm):
        return mm / pubfig.MM_PER_INCH
    
    @property
    def height(self):
        return self.width * self.height_u
    
    def add_child(self, lbwh=None, label=None, anchor=None, xy=None, wh=None,comment='...'):
        """
        Add a child panel and return it
        
        Parameters
        ----------
        lbwh : array-like, optional
            [left, bottom, width, height] in mm
        label : str, optional
            Label for the panel
        anchor : str or tuple, optional
            Anchor point for positioning when using xy + wh method.
            Options: 'center', 'top', 'bottom', 'left', 'right',
                    'top_left', 'top_right', 'bottom_left', 'bottom_right'
                    or (x_frac, y_frac) where 0 <= frac <= 1
        xy : array-like, optional
            [x, y] position of the anchor point in mm
        wh : array-like, optional
            [width, height] in mm
        
        Returns
        -------
        child : PanelChild
            The created child panel
        
        Examples
        --------
        # Method 1: Direct lbwh specification
        panel1 = fig.add_child([10, 10, 50, 60], label='A')
        
        # Method 2: Using anchor + xy + wh
        panel2 = fig.add_child(anchor='center', xy=[50, 50], wh=[40, 30], label='B')
        panel3 = fig.add_child(anchor='top_left', xy=[10, 100], wh=[40, 30], label='C')
        """
        if lbwh is not None:
            # Method 1: Direct lbwh specification
            child = PanelChild(self, lbwh, label,comment=comment)
        elif anchor is not None and xy is not None and wh is not None:
            # Method 2: Using anchor + xy + wh
            xy = np.asarray(xy)
            wh = np.asarray(wh)
            
            # Parse anchor to get fractional position
            if isinstance(anchor, tuple):
                x_frac, y_frac = anchor
            else:
                anchor_map = {
                    'center': (0.5, 0.5),
                    'top': (0.5, 1.0),
                    'bottom': (0.5, 0.0),
                    'left': (0.0, 0.5),
                    'right': (1.0, 0.5),
                    'top_left': (0.0, 1.0),
                    'top_right': (1.0, 1.0),
                    'bottom_left': (0.0, 0.0),
                    'bottom_right': (1.0, 0.0),
                }
                if anchor not in anchor_map:
                    raise ValueError(f"Unknown anchor: {anchor}")
                x_frac, y_frac = anchor_map[anchor]
            
            # Calculate left, bottom from anchor point
            left = xy[0] - wh[0] * x_frac
            bottom = xy[1] - wh[1] * y_frac
            
            lbwh = [left, bottom, wh[0], wh[1]]
            child = PanelChild(self, lbwh, label,comment=comment)
        else:
            raise ValueError("Either provide 'lbwh' or all of 'anchor', 'xy', and 'wh'")
        
        self.children.append(child)
        return child
    
    def get_child(self, identifier):
        """
        Get child by index or label
        
        Parameters
        ----------
        identifier : int or str
            If int: index of the child
            If str: label of the child
        
        Returns
        -------
        child : PanelChild
            The requested child panel
        
        Raises
        ------
        IndexError
            If index is out of range
        ValueError
            If label is not found
        """
        if isinstance(identifier, int):
            return self.children[identifier]
        elif isinstance(identifier, str):
            for child in self.children:
                if child.label == identifier:
                    return child
            raise ValueError(f"No child found with label: {identifier}")
        else:
            raise TypeError(f"identifier must be int or str, not {type(identifier)}")
    
    def plot_layout(self):
        """Plot layout for all children"""
        for child in self.children:
            child.plot_layout()
        plt.show()





if __name__ == '__main__':
    fig1 = pubfig('2col',.8)
    REDUCE_FACTOR = 0.01
    rf = REDUCE_FACTOR
    bl = fig1.add_child([0,.5,.9,.3],label ='a',
                        comment='<schematic of DW>\n' \
                        'crypy: lattice+triangleCDW+DW\n' \
                        'annotation: lattice unitcell+CDW unitcell + visualize relative shift +\n' \
                        '3Q direction+phase&phaseshift value')
    bl.reduce(w_reduce=rf,h_reduce=rf,anchor='left')
    xy = bl.get_point('bottom_right')
    
    
    #fig1.get_child(-1)
    fig1.add_child(label ='a_0',wh=[.2,.2],xy=xy,anchor='bottom_right',
                   comment='<orderparameter space diagram>\n' \
                   '')
    fig1.add_child([.9,.7,.1,.1],label ='a_1',comment='<cartoon of r-type dw>')
    fig1.add_child([.9,.6,.1,.1],label ='a_2')
    fig1.add_child([.9,.5,.1,.1],label ='a_3')

    ax = fig1.add_child([0,0,.5,.5],'b',comment='<topo of DW>\n' \
    'scalebar + nesting box of inset')
    ax.reduce(w_reduce=rf,h_reduce=rf,anchor='bottom_left')
    xy = ax.get_point('top_right')
    fig1.add_child(label = 'b_0',xy=xy,wh=[.2,.2],anchor='top_right',comment='<topo zoom>')

    ax = fig1.add_child([.5,0,.5,.5],'c',comment='<DWN phase map>\n' \
    'nesting box of b, scalebar')
    ax.reduce(w_reduce=rf,h_reduce=rf,anchor='bottom_right')     
    xy = ax.get_point('bottom_left')
    fig1.add_child(label = 'c_0',xy=xy,wh=[.2,.2],anchor='bottom_left',comment='<diagram of DWN>')
    
    

    # fig2 = pubfig('2col',.5)
    # fig2.add_child([.85,.5,.15,.15])

    fig1.plot_layout()

    # fig2.plot_layout()
    
