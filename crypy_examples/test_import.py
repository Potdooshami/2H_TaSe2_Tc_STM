import crypy as cp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches  # patches 모듈을 임포트
from math import gcd
from shapely.geometry import Point, Polygon

from crypy_examples.pts_on_poly import *

def draw_atom(x, y, radius=0.4, color_hex='#4169E1'):
    """
    그래데이션 효과가 적용된 단일 원자를 그리는 함수입니다. (수정 없음)
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
        circle = plt.Circle((x, y), radius * (1 - factor * 0.5), color=(r, g, b), zorder=10)
        ax.add_artist(circle)
    highlight = plt.Circle((x - radius * 0.2, y + radius * 0.2), radius * 0.2, color='white', alpha=0.5, zorder=11)
    ax.add_artist(highlight)

def draw_bond(x, y, r, **kwargs):
    """
    matplotlib.patches.Rectangle을 사용하여 두 점 (x1, y1)과 (x2, y2) 사이에
    데이터 좌표 기준 폭 r을 가진 사각형(bond)을 그립니다.

    매개변수:
    x (list-like): (x1, x2) 형태의 x좌표 시퀀스
    y (list-like): (y1, y2) 형태의 y좌표 시퀀스
    r (float):     사각형의 폭 (데이터 좌표 기준)
    **kwargs:      Rectangle 패치에 전달할 추가 스타일 인자 
                   (예: facecolor='blue', alpha=0.5 등)
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

def hex_boundary(n):
    if n == 0:
        return np.array([[0,0]])
    else:
        ks = np.arange(n)
        ks = ks.reshape(1,1,-1)

        n_holder = np.array([[0,1],[1,1],[1,0]]).transpose()
        k_holder = np.array([[-1,-1],[-1,0],[0,1]]).transpose()
        # n_holder = np.array([[1,0],[1,1],[1,0]]).transpose()
        # k_holder = np.array([[-1,-1],[-1,0],[0,1]]).transpose()
        n_part =   n*n_holder[:,:,np.newaxis]
        k_part = ks*k_holder[:,:,np.newaxis]
        foo = n_part + k_part
        foo = foo.reshape(2,-1)
        foo = np.concatenate((foo,-foo),axis=1)
        foo = foo.transpose()

        return foo

def hex_closure(n):
    """
    hex_boundary(0)부터 hex_boundary(n)까지의 모든 배열을
    누적하여 하나의 배열로 반환합니다.
    """
    # 0부터 n까지 각 "경계" (boundary)를 계산하여 리스트에 담습니다.
    all_boundaries = [hex_boundary(i) for i in range(n + 1)]
    
    # 리스트에 담긴 모든 (k, 2) 형태의 배열들을
    # 세로 방향(axis=0)으로 연결하여 하나의 (m, 2) 배열로 만듭니다.
    return np.concatenate(all_boundaries, axis=0)
def gen_hex(R):
    tht = 2*np.pi*(1/12+np.arange(0,6)/6)
    v_xy =R*(2/np.sqrt(3))*np.array([np.cos(tht),np.sin(tht)])
    polygon = v_xy.transpose()
    return polygon

def lattice_points_in_hex(a1, a2,R ,contain_boundary=True):
    polygon = gen_hex(R)
    if contain_boundary:
        R = R +0.01
    else:
        R = R -0.01
    foo,indices = lattice_points_in_polygon(a1, a2, polygon)
    return indices
#-----main--------------------------------------------------------------------------
a1 = [1,0]
a2 = [-0.5,3**0.5/2]
p1 = np.array((2,1))/3
p2 = np.array((1,2))/3
p3=np.array((1,-1))/3 
p4=-p3

n_dom =6
n_supsup = 2*3*n_dom-2
domain_range =  ((-1,1),(-1,1))
n__ = 2*n_supsup
latt_range = ((-n__,n__),(-n__,n__))
#------class declare-----------------------------------------------------------


pv = cp.PrimitiveVector2D(a1,a2)
bss = cp.Basis2D(pv)
gen_atom_Ta = lambda x,y: draw_atom(x, y, radius=0.2, color_hex='#4169E1')
gen_atom_Se = lambda x,y: draw_atom(x, y, radius=0.1, color_hex='#FF6347')
gen_bond = lambda x,y: draw_bond(x,y,r=.05,facecolor=(.7,.7,.7))


bss.add_artist(gen_atom_Ta,(p1),label='Ta')
bss.add_artist(gen_atom_Se,(p2),label='Se')
bss.add_artist(gen_bond,(p1,p2),label = 'bond1')
bss.add_artist(gen_bond,(p1,p3),label = 'bond2')
bss.add_artist(gen_bond,(p2,p4),label = 'bond3')
# xylim = ((-10,10),(-10,10))

lp = cp.LatticePoints2D(pv) 
# lp.generate_points_by_xylim(*xylim)
lp.generate_points_by_range(*latt_range)
cry = cp.Crystal2D(bss,lp)

pv_sup = pv.get_super_structure(3,3)
bss_sup = cp.Basis2D(pv_sup)
gen_CDW = lambda xxx,yyy: plt.fill(xxx,yyy,"y",alpha=.3)
# cdw = dict()
cdw_p1 = (2/3,1/3)
cdw_p2 = (-1/3,1/3)
cdw_p3 = (-1/3,-2/3)
cdw_ps = (cdw_p1,cdw_p2,cdw_p3)
bss_sup.add_artist(gen_CDW,cdw_ps,label='CDW')





lp_sup = cp.LatticePoints2D(pv_sup) 
# lp_sup.generate_points_by_xylim(*xylim)


ps = lattice_points_in_hex(a1, a2,n_dom)
lp_sup.generate_points_by_manual(ps)
cry_sup = cp.Crystal2D(bss_sup,lp_sup)
gen_domain = lambda x,y:cry_sup.plot_crystal(x,y)

pv_supsup = pv.get_super_structure(n_supsup,n_supsup)
bss_supsup = cp.Basis2D(pv_supsup)
bss_supsup.add_artist(gen_domain,(0,0),label='domain')
lp_supsup =  cp.LatticePoints2D(pv_supsup)
lp_supsup.generate_points_by_range(*domain_range)
cry_supsup = cp.Crystal2D(bss_supsup,lp_supsup)
cry_supsup.plot_crystal()
# cry.plot_crystal()


#----------------------------------------------------

#cry_sup._basis.plot_basis()
# cry_sup._basis.primitive_vector.plot_all()
#cry_sup._lattice.primitive_vector.plot_all()

# cry_sup._lattice.plot_scatter()

# fig,ax  = cry.plot_crystal()
# ax.set_xlim(-8,8)
# ax.set_ylim(-8,8)


plt.show()


#----------------------------------------------------
