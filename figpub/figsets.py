import os
from matplotlib import pyplot as plt
import numpy as np


class PanelChild:
    """Individual panel within a publication figure"""
    
    # --- 1. 'draw' 인자 추가, ax 생성 코드 제거 ---
    def __init__(self, parent_fig, lbwh, label=None, comment='...', draw=None): # <--- draw 추가
        """
        Parameters
        ----------
        parent_fig : PubFig
            Parent figure object
        lbwh : array-like
            [left, bottom, width, height] in mm
        label : str, optional
            Label for the panel
        draw : callable, optional
            Function(ax) to draw on this panel
        """
        self.parent = parent_fig
        self.lbwh = np.asarray(lbwh, dtype=float)
        self.label = label if label is not None else 'unknown'
        self.comment = comment
        self.draw = draw       # <--- draw 함수 핸들 저장
        self.ax = None         # <--- ax를 None으로 초기화 (지연 초기화)
        
        # --- 삭제됨 ---
        # lbwh_axu = ...
        # self.ax = parent_fig.fig.add_axes(lbwh_axu)
    
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
    
    # --- 2. render 메서드 신설 ---
    def render(self):
        """
        (PubFig.render()에 의해 호출됨)
        실제 Matplotlib Axes 객체를 생성합니다.
        """
        if self.ax is None: # <--- 아직 렌더링되지 않았다면
            if self.parent.fig is None:
                # 부모가 먼저 렌더링되어야 함
                self.parent.render()
                
            # 부모의 종횡비(height_u)를 사용해 정규화된 좌표 계산
            lbwh_axu = self.lbwh * np.array([1, 1/self.parent.height_u, 1, 1/self.parent.height_u])
            self.ax = self.parent.fig.add_axes(lbwh_axu)

    def resize(self, width=None, height=None, anchor='bottom_left'):
        # ... (내부 로직은 동일) ...
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
        
        self._update_axes() # <--- _update_axes 호출
    
    def reduce(self, w_reduce=0, h_reduce=0, anchor='bottom_left'):
        # ... (내부 로직은 동일) ...
        new_width = self.width - w_reduce
        new_height = self.height - h_reduce
        self.resize(width=new_width, height=new_height, anchor=anchor)
    
    def translate(self, dx=0, dy=0):
        # ... (내부 로직은 동일) ...
        self.lbwh[0] += dx
        self.lbwh[1] += dy
        self._update_axes()

    def set_position(self, left=None, bottom=None):
        # ... (내부 로직은 동일) ...
        if left is not None:
            self.lbwh[0] = left
        if bottom is not None:
            self.lbwh[1] = bottom
        self._update_axes()
    
    def get_point(self, anchor='center'):
        # ... (변경 없음) ...
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
    
    # --- 3. _update_axes 수정 ---
    def _update_axes(self):
        """Update axes position after transformation"""
        if self.ax is not None: # <--- 렌더링된 후에만 작동
            lbwh_axu = self.lbwh * np.array([1, 1/self.parent.height_u, 1, 1/self.parent.height_u])
            self.ax.set_position(lbwh_axu)
    
    # --- 4. plot_layout 수정 ---
    def plot_layout(self):
        """Draw layout helper (cross and label)"""
        if self.ax is None: # <--- 렌더링이 필요하면 자동 렌더링
            self.render()
            
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

    # --- 5. plot_draw 메서드 신설 ---
    def plot_draw(self):
        """
        지정된 'draw' 함수를 사용해 실제 데이터를 플롯합니다.
        draw 함수가 없으면 layout 헬퍼를 대신 그립니다.
        """
        if self.ax is None: # <--- 렌더링이 필요하면 자동 렌더링
            self.render()
            
        if self.draw is not None:
            self.draw(self.ax) # <--- 저장된 함수를 ax 인자와 함께 호출
        else:
            # draw 함수가 없으면 대신 레이아웃 표시
            self.plot_layout()


class PubFig:
    MM_PER_INCH = 25.4
    WIDTH_2COL = 178
    WIDTH_1COL = 86
    
    # --- 1. __init__ 수정 ---
    def __init__(self, width, height_u,width_rescale = 1):
        if width == '1col':
            self.width_pure = PubFig.WIDTH_1COL
        elif width == '2col':
            self.width_pure = PubFig.WIDTH_2COL
        else:
            self.width_pure = width
            
        self.width = self.width_pure * width_rescale
        self.height_u = height_u
        self.fig = None        # <--- fig를 None으로 초기화 (지연 초기화)
        self.fignum = None     # <--- fignum도 None으로 초기화
        self.children = []
        
        # --- 삭제됨 ---
        # self.fig = plt.figure(...)

    @staticmethod
    def mm_to_inch(mm):
        return mm / PubFig.MM_PER_INCH
    
    @property
    def height(self):
        return self.width * self.height_u
    
    # --- 2. render 메서드 신설 ---
    def render(self):
        """
        (PubProject에 의해 호출됨)
        실제 Matplotlib Figure 객체를 생성하고,
        모든 자식 패널의 렌더링을 트리거합니다.
        """
        if self.fig is None: # <--- 아직 렌더링되지 않았다면
            figsize = (PubFig.mm_to_inch(self.width), 
                       PubFig.mm_to_inch(self.height))
            self.fig = plt.figure(figsize=figsize)
            self.fignum = self.fig.number
            
            # 모든 자식 패널도 렌더링
            for child in self.children:
                child.render()
    
    # --- 3. add_child 수정 ---
    def add_child(self, lbwh=None, label=None, anchor=None, xy=None, wh=None, comment='...', draw=None): # <--- draw 추가
        """
        ... (docstring은 동일) ...
        """
        if lbwh is not None:
            # Method 1: Direct lbwh specification
            child = PanelChild(self, lbwh, label, comment=comment, draw=draw) # <--- draw 전달
        elif anchor is not None and xy is not None and wh is not None:
            # Method 2: Using anchor + xy + wh
            xy = np.asarray(xy)
            wh = np.asarray(wh)
            
            # ... (anchor_map 로직 동일) ...
            if isinstance(anchor, tuple):
                x_frac, y_frac = anchor
            else:
                anchor_map = {
                    'center': (0.5, 0.5), 'top': (0.5, 1.0), 'bottom': (0.5, 0.0),
                    'left': (0.0, 0.5), 'right': (1.0, 0.5), 'top_left': (0.0, 1.0),
                    'top_right': (1.0, 1.0), 'bottom_left': (0.0, 0.0), 'bottom_right': (1.0, 0.0),
                }
                if anchor not in anchor_map:
                    raise ValueError(f"Unknown anchor: {anchor}")
                x_frac, y_frac = anchor_map[anchor]
            
            # Calculate left, bottom from anchor point
            left = xy[0] - wh[0] * x_frac
            bottom = xy[1] - wh[1] * y_frac
            
            lbwh = [left, bottom, wh[0], wh[1]]
            child = PanelChild(self, lbwh, label, comment=comment, draw=draw) # <--- draw 전달
        else:
            raise ValueError("Either provide 'lbwh' or all of 'anchor', 'xy', and 'wh'")
        
        self.children.append(child)
        return child
    
    def get_child(self, identifier):
        # ... (변경 없음) ...
        if isinstance(identifier, int):
            return self.children[identifier]
        elif isinstance(identifier, str):
            for child in self.children:
                if child.label == identifier:
                    return child
            raise ValueError(f"No child found with label: {identifier}")
        else:
            raise TypeError(f"identifier must be int or str, not {type(identifier)}")
    
    # --- 4. plot_layout 수정 ---
    def plot_layout(self):
        """Plot layout for all children"""
        if self.fig is None: # <--- 렌더링이 필요하면 자동 렌더링
            self.render()
            
        for child in self.children:
            child.plot_layout()
        # plt.show()
        

class PubProject:
    """
    여러 개의 PubFig 객체를 하나의 프로젝트로 관리합니다.
    (예: 논문 하나에 포함된 Fig 1, Fig 2, Fig 3...)
    """
    
    def __init__(self, *figs):
        # ... (변경 없음) ...
        self.figs = []
        for fig in figs:
            if not isinstance(fig, PubFig):
                raise TypeError(f"모든 인자는 PubFig 객체여야 합니다. {type(fig)} 타입이 입력되었습니다.")
            self.figs.append(fig)
        
        print(f"PubProject가 {len(self.figs)}개의 Figure로 생성되었습니다.")

    def __getitem__(self, index):
        # ... (변경 없음) ...
        return self.figs[index]

    def __len__(self):
        # ... (변경 없음) ...
        return len(self.figs)

    # --- 1. plot_layouts 수정 ---
    def plot_layouts(self):
        """프로젝트에 포함된 모든 Figure의 레이아웃을 그립니다."""
        for i, pub_fig in enumerate(self.figs):
            try:
                pub_fig.render() # <--- 여기서 명시적으로 렌더링
                pub_fig.plot_layout()
                
                # 창 위치 이동 (렌더링 후에만 가능)
                manager = plt.get_current_fig_manager()
                x_pos = 500 + i * 200
                y_pos = 100 + i * 50
                manager = pub_fig.fig.canvas.manager
                manager.window.move(x_pos, y_pos)
            except Exception as e:
                print(f"Figure {i} 레이아웃 플롯팅 중 오류 발생: {e}")

    # --- 2. plot_draws 메서드 신설 ---
    def plot_draws(self):
        """프로젝트에 포함된 모든 Figure의 실제 데이터 플롯을 그립니다."""
        for i, pub_fig in enumerate(self.figs):
            try:
                pub_fig.render() # <--- 여기서 명시적으로 렌더링
                
                # 모든 자식의 plot_draw 호출
                for child in pub_fig.children:
                    child.plot_draw()
                    
                # 창 위치 이동 (렌더링 후에만 가능)
                manager = plt.get_current_fig_manager()
                x_pos = 500 + i * 200
                y_pos = 100 + i * 50
                manager = pub_fig.fig.canvas.manager
                manager.window.move(x_pos, y_pos)
            except Exception as e:
                print(f"Figure {i} 플롯팅 중 오류 발생: {e}")
                
    # --- 3. show 수정 ---
    def show(self):
        """plt.show()를 호출하여 모든 Figure를 한꺼번에 보여줍니다."""
        # 이제 이 메서드는 순수하게 plt.show()만 호출합니다.
        # (렌더링은 plot_draws나 plot_layouts가 담당)
        plt.show()

    # --- 4. save_all 수정 ---
    def save_all(self, directory='.', prefix='Fig', format='pdf', dpi=300, **kwargs):
        """
        ... (docstring은 동일) ...
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"디렉토리를 생성했습니다: {directory}")
        
        for i, pub_fig in enumerate(self.figs):
            filename = os.path.join(directory, f"{prefix}{i+1}.{format}")
            try:
                pub_fig.render() # <--- 저장 직전에 렌더링
                
                # PubFig 객체는 .fig 속성으로 matplotlib Figure 객체를 가집니다.
                pub_fig.fig.savefig(filename, format=format, dpi=dpi, **kwargs)
                print(f"저장 완료: {filename}")
            except Exception as e:
                print(f"Figure {i} 저장 중 오류 발생 ({filename}): {e}")


# --- 5. __main__ 블록 수정 (새로운 워크플로우 시연) ---
if __name__ == '__main__':

    # --- 플로팅 함수 정의 (예시) ---
    def draw_a(ax):
        ax.set_title("Panel A - Data")
        ax.plot([1, 2, 3], [3, 1, 5], 'r-o')

    def draw_b(ax):
        ax.set_title("Panel B - Image")
        ax.imshow(np.random.rand(10, 10))

    # --- fig1 생성 (이때는 Figure 객체 안 만들어짐) ---
    fig1 = PubFig('2col',.8,width_rescale=.8)
    REDUCE_FACTOR = 0.01
    rf = REDUCE_FACTOR
    
    # add_child에 draw 함수 연결
    bl = fig1.add_child([0,.5,.9,.3],label ='a',
                        comment='<schematic of DW>',
                        draw=draw_a) # <--- draw_a 함수 연결
    bl.reduce(w_reduce=rf,h_reduce=rf,anchor='left')
    xy = bl.get_point('bottom_right')
    
    fig1.add_child(label ='a_0',wh=[.2,.2],xy=xy,anchor='bottom_right',
                   comment='<orderparameter space diagram>')
    fig1.add_child([.9,.7,.1,.1],label ='a_1',comment='<cartoon of r-type dw>')
    fig1.add_child([.9,.6,.1,.1],label ='a_2')
    fig1.add_child([.9,.5,.1,.1],label ='a_3')

    ax = fig1.add_child([0,0,.5,.5],'b',
                        comment='<topo of DW>',
                        draw=draw_b) # <--- draw_b 함수 연결
    ax.reduce(w_reduce=rf,h_reduce=rf,anchor='bottom_left')
    xy = ax.get_point('top_right')
    fig1.add_child(label = 'b_0',xy=xy,wh=[.2,.2],anchor='top_right',comment='<topo zoom>')

    ax = fig1.add_child([.5,0,.5,.5],'c',comment='<DWN phase map>')
    ax.reduce(w_reduce=rf,h_reduce=rf,anchor='bottom_right')    
    xy = ax.get_point('bottom_left')
    fig1.add_child(label = 'c_0',xy=xy,wh=[.2,.2],anchor='bottom_left',comment='<diagram of DWN>')

    # --- fig2 생성 ---
    fig2 = PubFig('2col',.5)
    # ... (fig2.add_child 코드들 - 생략) ...

    # --- PubProject 생성 ---
    # fig1, fig2 객체를 프로젝트에 등록
    project = PubProject(fig1, fig2) 

    # --- 실행 ---
    # 1. 레이아웃만 확인하고 싶을 때:
    # project.plot_layouts() 
    
    # 2. 실제 데이터로 그리고 싶을 때:
    # (이때 fig1, fig2의 render()가 호출되고, 
    #  'a', 'b' 패널은 draw 함수가, 나머지는 plot_layout이 실행됨)
    project.plot_draws() 

    # 3. 창 띄우기
    project.show()
    
    # 4. 저장하기
    # project.save_all(directory='my_figures', format='pdf')