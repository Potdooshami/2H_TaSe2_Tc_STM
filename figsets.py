import os
from matplotlib import pyplot as plt
import numpy as np



class PanelChild:
    """Individual panel within a publication figure"""
    
    def __init__(self, parent_fig, lbwh, label=None,comment='...'):
        """
        Parameters
        ----------
        parent_fig : PubFig
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


class PubFig:
    MM_PER_INCH = 25.4
    WIDTH_2COL = 178
    WIDTH_1COL = 86
    
    def __init__(self, width, height_u,width_rescale = 1):        
        if width == '1col':
            self.width_pure = PubFig.WIDTH_1COL
        elif width == '2col':
            self.width_pure = PubFig.WIDTH_2COL
        else:
            self.width_pure = width
            
        self.width = self.width_pure * width_rescale
        self.height_u = height_u            
        self.fig = plt.figure(figsize=(PubFig.mm_to_inch(self.width), 
                                       PubFig.mm_to_inch(self.height)))
        self.children = []
    
    @staticmethod
    def mm_to_inch(mm):
        return mm / PubFig.MM_PER_INCH
    
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
        # plt.show()    
class PubProject:
    """
    여러 개의 PubFig 객체를 하나의 프로젝트로 관리합니다.
    (예: 논문 하나에 포함된 Fig 1, Fig 2, Fig 3...)
    """
    
    def __init__(self, *figs):
        """
        프로젝트를 초기화합니다.
        
        Parameters
        ----------
        *figs : PubFig
            프로젝트에 포함할 PubFig 객체들을 가변 인자로 받습니다.
        """
        self.figs = []
        for fig in figs:
            if not isinstance(fig, PubFig):
                raise TypeError(f"모든 인자는 PubFig 객체여야 합니다. {type(fig)} 타입이 입력되었습니다.")
            self.figs.append(fig)
        
        print(f"PubProject가 {len(self.figs)}개의 Figure로 생성되었습니다.")

    def __getitem__(self, index):
        """인덱스를 사용해 특정 Figure에 접근할 수 있게 합니다 (예: project[0])."""
        return self.figs[index]

    def __len__(self):
        """프로젝트에 포함된 Figure의 개수를 반환합니다 (예: len(project))."""
        return len(self.figs)

    def plot_layouts(self):
        """프로젝트에 포함된 모든 Figure의 레이아웃을 그립니다."""
        for i, pub_fig in enumerate(self.figs):
            try:
                pub_fig.plot_layout()
            except Exception as e:
                print(f"Figure {i} 레이아웃 플롯팅 중 오류 발생: {e}")

    def show(self):
        """plt.show()를 호출하여 모든 Figure를 한꺼번에 보여줍니다."""
        plt.show()

    def save_all(self, directory='.', prefix='Fig', format='pdf', dpi=300, **kwargs):
        """
        프로젝트의 모든 Figure를 지정된 디렉토리에 저장합니다.
        
        Parameters
        ----------
        directory : str, optional
            Figure를 저장할 디렉토리. 기본값: '.' (현재 디렉토리)
        prefix : str, optional
            저장할 파일명의 접두사 (예: 'Fig' -> Fig1.pdf, Fig2.pdf ...). 
            기본값: 'Fig'
        format : str, optional
            파일 포맷 (예: 'pdf', 'png', 'svg'). 기본값: 'pdf'
        dpi : int, optional
            PNG, JPG 등 래스터 이미지의 해상도. 기본값: 300
        **kwargs
            plt.Figure.savefig()에 전달할 추가 키워드 인자
            (예: bbox_inches='tight')
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"디렉토리를 생성했습니다: {directory}")
        
        for i, pub_fig in enumerate(self.figs):
            filename = os.path.join(directory, f"{prefix}{i+1}.{format}")
            try:
                # PubFig 객체는 .fig 속성으로 matplotlib Figure 객체를 가집니다.
                pub_fig.fig.savefig(filename, format=format, dpi=dpi, **kwargs)
                print(f"저장 완료: {filename}")
            except Exception as e:
                print(f"Figure {i} 저장 중 오류 발생 ({filename}): {e}")




if __name__ == '__main__':
    fig1 = PubFig('2col',.8,width_rescale=.8)
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
#----------------------------------------------------------------------------------------------------------------
    fig2 = PubFig('2col',.5)
    lu = .25
    ls = [lu, lu/2,.2, .8-lu*(3/2)]
    xs = []
    ys = [0,.25]
    
    fig2.add_child([0,0,ls[-1],.5],label='a',comment='<schematic of vortex>').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
    fig2.add_child([ls[-1],lu,lu,lu],label='b',comment='<topo of R vortex>').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
    fig2.add_child([ls[-1]+lu,lu*(3/2),lu/2,lu/2],label='c',comment='<shiftmap of R vortex>').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
    fig2.add_child([ls[-1]+lu,lu*(1),lu/2,lu/2],label='d',comment='<tripletmap of R vortex>').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
    fig2.add_child([ls[-1],0,lu,lu],label='e',comment ='<... of L>').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
    fig2.add_child([ls[-1]+lu,lu*(1/2),lu/2,lu/2],label='f').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
    fig2.add_child([ls[-1]+lu,lu*(0),lu/2,lu/2],label='g').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
    fig2.add_child([ls[-1]+lu*(3/2),0,.2,.5],label='h',
                   comment='<schematic of vortex joining>\n' \
                   'R,L vortex\n' \
                   '3 R-L bonding\n' \
                   'honeycomb tile').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
#----------------------------------------------------------------------------------------------------------------    
    h1 = .1 
    h2 =  .3
    h3 = .3
    hs = [h1,h2,h3]
    h123 = h1+h2+h3
    fig3 = PubFig('1col',1+h123)    
    fig3.add_child([0,h123,1,1],label='a',
                   comment='<schematic of DWN>\n' \
                   'background white with phase value\n' \
                   'directional DW + vortex symbol\n' \
                   'bounding Loop set1: unique boundary decomposition,\n' \
                   'bounding Loop set2: Vortex calculation').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
    lbls = 'bcd'
    ys = [h2+h3,h3,0]
    sublbls =['','_1','_2']
    w = 1/3
    for iloop,lbl_main in zip(range(3),lbls):
        x= iloop/3
        for iinfo,lbl_sub,y,h in zip(range(3),sublbls,ys,hs):
            xywh =[x,y,w,h]
            label = lbl_main+lbl_sub
            fig3.add_child(xywh,label=label).reduce(w_reduce=rf,h_reduce=rf,anchor='center')
    fig3.get_child('b').comment = '<pathwords>:boundcary decomposition'
    fig3.get_child('b_1').comment = '<geometric>'
    fig3.get_child('b_2').comment = '<algebraic>'
    fig3.get_child('c').comment = 'charge calculation 1'
    fig3.get_child('d').comment = 'charge calculation 2'

    
#----------------------------------------------------------------------------------------------------------------
    h1 = .7
    hk23 = .15
    fig4 = PubFig('1col',1+ h1)
    alikes ='ac'
    blikes ='bd'
    xs =[0,.5]
    for ind,alike,blike,x in zip(range(2),alikes,blikes,xs):
        ax = fig4.add_child([x,.5+h1,.5,.5],label=alike)
        ax.reduce(w_reduce=rf,h_reduce=rf,anchor='center')
        xy =ax.get_point('top_right')
        fig4.add_child(xy=xy,wh=[.2,.2],anchor='top_right',label= alike +'_1')    
        ax = fig4.add_child([x,+h1,.5,.5],label=blike)
        ax.reduce(w_reduce=rf,h_reduce=rf,anchor='center')
        xy =ax.get_point('top_right')
        ax = fig4.add_child(xy=xy,wh=[hk23,hk23],anchor='top_right',label=blike+ '_2')
        xy =ax.get_point('top_left')
        fig4.add_child(xy=xy,wh=[hk23,hk23],anchor='top_right',label=blike+ '_1')
    fig4.add_child([0,0,1,h1],label='e',
                comment = '<cartoon of phase pinning>\n' \
                '2 layer of projected schematic. upper(lower) is IC(C)\n' \
                'left-side: arrow with text phase-pinning\n' \
                'right-side: 1d version of pinnint').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
    fig4.get_child('a').comment = '<topo-78K>'
    fig4.get_child('a_1').comment = '<FFT-78K>\nhighlight Q peaks'
    fig4.get_child('b').comment = '<phase-78K>\nk1'
    fig4.get_child('b_1').comment = 'k2'
    fig4.get_child('b_2').comment = 'k3'
    fig4.get_child('c').comment = '...-110K'


    fig5 = PubFig(820,.2)

#----------------------------------------------------------------------------------------------------------------
    fig1.plot_layout()
    fig2.plot_layout()
    fig3.plot_layout()
    fig4.plot_layout()    
    plt.show()
    
