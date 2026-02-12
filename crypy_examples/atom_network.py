import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
import crypy as cp

a1 = [1,0] # lattic unitcell vector 1
a2 = [-0.5,3**0.5/2] # lattic unitcell vector 2
p1 = np.array((2,1))/3 # basis point1
p2 = np.array((1,2))/3 # basis point2
p3=np.array((1,-1))/3  # basis point3
p4=-p3 # basis point4

color_bond="#E0E0E0FF"
color_Se ="#FF6347"
color_Ta='#4169E1'
r_Se = .1
r_Ta = .2
r_bond = .1


def draw_atom(x, y, radius=0.4, color_hex='#4169E1'):
    """ draw an atom

    gradient sphere exture ball    

    Args:
        x (float): x coordinate of the atom center
        y (float): y coordinate of the atom center
        radius (float, optional): radius of the atom. Defaults to 0.4.
        color_hex (str, optional): color of the atom in hex format. Defaults to '#4169E1'.
        
    Returns:
        None
    """
    ax = plt.gca()
    for i in range(10):
        factor = i / 10.0
        r_base = int(color_hex[1:3], 16) / 255.0
        g_base = int(color_hex[3:5], 16) / 255.0
        b_base = int(color_hex[5:7], 16) / 255.0
        r = r_base * (1 - factor) + factor * 0.8
        g = g_base * (1 - factor) + factor * 0.8
        b = b_base * (1 - factor) + factor * 0.8
        r, g, b = max(0, min(r, 1)), max(0, min(g, 1)), max(0, min(b, 1))
        circle = plt.Circle((x, y), radius * (1 - factor * 0.5), color=(r, g, b))
        ax.add_artist(circle)
    highlight = plt.Circle((x - radius * 0.2, y + radius * 0.2), radius * 0.2, color='white', alpha=0.5)
    ax.add_artist(highlight)

def draw_bond(x, y, r, **kwargs):
    """ draw a bond between two points using a rectangle patch

    draw rectangle between (x1, y1) and (x2, y2)

    Args:
        x (list-like): (x1, x2) 형태의 x좌표 시퀀스
        y (list-like): (y1, y2) 형태의 y좌표 시퀀스
        r (float):     사각형의 폭 (데이터 좌표 기준)
        **kwargs:      Rectangle 패치에 전달할 추가 스타일 인자 (예: facecolor='blue', alpha=0.5 등)
    Returns:
        None
    """
    
    # 1. 현재 축(axes) 가져오기
    # 함수 외부에서 `ax = plt.subplot()` 등으로 축을 미리 만들어 두는 것이 좋습니다.
    # 여기서는 편의상 plt.gca() (Get Current Axes)를 사용합니다.
    ax = plt.gca()

    # 2. 좌표 및 벡터 계산
    x1, x2 = x
    y1, y2 = y
    
    dx = x2 - x1
    dy = y2 - y1

    # 3. 본드 길이(length) 및 각도(angle) 계산
    # np.hypot은 sqrt(dx**2 + dy**2) 보다 수치적으로 안정적입니다.
    length = np.hypot(dx, dy) 
    
    # 0 길이의 본드는 그리지 않음 (오류 방지)
    if length == 0:
        return

    # x축 양의 방향 기준 반시계 방향 각도 (라디안)
    angle_rad = np.arctan2(dy, dx)
    # matplotlib patch는 '도' (degree) 단위를 사용
    angle_deg = np.degrees(angle_rad) 

    # 4. 사각형의 앵커(anchor) 위치 계산
    # Rectangle은 (x, y) 앵커를 '회전 전' 좌하단 모서리로 하여 angle만큼 회전합니다.
    # (x1, y1)에서 (x2, y2) 라인을 중심으로 하려면,
    # 앵커를 (x1, y1)에서 선의 수직 방향으로 -r/2 만큼 이동시켜야 합니다.
    
    # (x1, y1) 기준, 수직(반시계 90도) 방향 단위 벡터: (-dy/length, dx/length)
    # 우리가 필요한 이동 방향 (-r/2, 즉 시계방향 90도): (dy/length, -dx/length)
    
    shift_x = (dy / length) * (r / 2)
    shift_y = (-dx / length) * (r / 2)
    
    anchor_x = x1 + shift_x
    anchor_y = y1 + shift_y

    # 5. 스타일(kwargs) 설정
    # 'facecolor'가 제공되지 않으면 'black'을 기본값으로 사용
    if 'facecolor' not in kwargs:
        kwargs['facecolor'] = 'black'
    # 'edgecolor'를 없애면 경계선이 사라져 더 깔끔해 보일 수 있습니다.
    if 'edgecolor' not in kwargs:
        kwargs['edgecolor'] = 'none' # 또는 kwargs['facecolor']

    # 6. Rectangle 패치 생성
    rect = patches.Rectangle(
        (anchor_x, anchor_y),   # (x, y) 앵커 (회전 전 좌하단)
        length,                 # width (본드 길이)
        r,                      # height (본드 폭)
        angle=angle_deg,        # 회전 각도 (도)
        **kwargs                # 스타일 인자 적용 (facecolor, alpha 등)
    )

    # 7. 현재 축(ax)에 패치 추가
    ax.add_patch(rect)
def draw_atom_simple(x,y,radius=0.4,color_hex='#4169E1'):
    cp.Collection.Generator.gen_regular_polygon(20,x,y,r=radius,c=color_hex)
gen_atom_Ta_v1 = lambda x,y: draw_atom(x, y, radius=r_Ta, color_hex=color_Ta)
gen_atom_Ta_v2 = lambda x,y: draw_atom_simple(x, y, radius=r_Ta, color_hex=color_Ta)
gen_atom_Ta_hidden = lambda x,y: draw_atom_simple(x, y, radius=r_Ta, color_hex=to_rgb(color_bond))
gen_atom_Se = lambda x,y: draw_atom(x, y, radius=r_Se, color_hex=color_Se)
gen_atom_Se_simple = lambda x,y: draw_atom_simple(x, y, radius=r_Se, color_hex=color_Se)
gen_atom_Se_hidden = lambda x,y: draw_atom_simple(x, y, radius=r_Se, color_hex=to_rgb(color_bond))
gen_bond = lambda x,y: draw_bond(x,y,r=r_bond,facecolor=to_rgb(color_bond))



color_Se_HA = [254,164,0]
color_Ta_HA = [0, 176, 240]
hexer =  lambda rgb: '#{:02x}{:02x}{:02x}'.format(*rgb)
gen_atom_Ta_vHA = lambda x,y: draw_atom(x, y, radius=r_Ta*.6*1.5, color_hex=hexer(color_Ta_HA))
gen_atom_Se_vHA = lambda x,y: draw_atom(x, y, radius=r_Se*1.5, color_hex=hexer(color_Se_HA))
gen_atom_Ta_vHA_simple = lambda x,y: draw_atom_simple(x, y, radius=r_Ta*.6*1.5, color_hex=hexer(color_Ta_HA))
gen_atom_Se_vHA_simple = lambda x,y: draw_atom_simple(x, y, radius=r_Se*1.5, color_hex=hexer(color_Se_HA))
gen_bond_vHA = lambda x,y: draw_bond(x,y,r=r_bond*.6,facecolor=to_rgb(color_bond))
