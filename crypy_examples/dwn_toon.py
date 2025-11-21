import numpy as np
import crypy as cp
import math
import matplotlib.pyplot as plt
from useful import fullax,savepng
CLR_EDGE = np.eye(3)
CLR_NODE = np.array([(255, 165, 0),(0, 127, 255)])/256
def gen_directional_line(xx,yy,c,n=10,sharpness=1,biasness=0):
    p12 = np.array((xx,yy))
    p1 = p12[:,0]
    p2 = p12[:,1]
    def get_perpendicular_unit_vector(v):
        """
        주어진 2D 벡터(v)에 수직인 단위 벡터를 반환합니다.
        v는 (x, y) 형태의 튜플이나 리스트일 수 있습니다.
        
        벡터 (x, y)에 대해 90도 반시계 방향으로 회전한
        수직 벡터 (-y, x)를 기반으로 단위 벡터를 계산합니다.
        """
        x, y = v
        
        # 벡터의 크기(magnitude) 계산
        magnitude = math.sqrt(x**2 + y**2)
        
        # 0 벡터 (Zero vector) 예외 처리
        if magnitude == 0:
            return (0.0, 0.0)
        
        # 수직 벡터 (-y, x)
        perp_x = -y
        perp_y = x
        
        # 수직 벡터를 정규화(normalize)하여 단위 벡터로 만듦
        unit_perp_x = perp_x / magnitude
        unit_perp_y = perp_y / magnitude
        
        return np.array((unit_perp_x, unit_perp_y))
    p1 = np.array(p1)
    p2 = np.array(p2)
    #-------------------------------------------

    d = p2-p1
    d_norm = np.linalg.norm(d)
    d_neck_norm = (1/(2*(3**.5)))*(d_norm/n)

    d_neck = sharpness*d_neck_norm*get_perpendicular_unit_vector(d)
    # for i_n in range(n):
    p_neck1 = p1-d_neck
    v_from_p_neck1_to_p_neck2 = d/n
    v_from_p_neck1_to_p_nose = 0.5*(d/n)+3*d_neck
    p_neck2 = p_neck1 + v_from_p_neck1_to_p_neck2
    p_nose = p_neck1 + v_from_p_neck1_to_p_nose
    p_tri = np.array((p_neck1,p_neck2,p_nose)) + d_neck*biasness
    for i_n in range(n):
        p_tri_now = p_tri + (d/n)*i_n
        plt.fill(p_tri_now[:,0],p_tri_now[:,1],c=c)

if __name__ == '__main__':    
    a1 = [1,0]
    a2 = [-0.5,3**0.5/2]
    pv_dwdv = cp.PrimitiveVector2D(a1,a2)
    pv_dwdv_sub = pv_dwdv.get_sub_structure(3,3) # unitcell 안에서 새부적인 작업을 하기 위해 # unitcell 안에서 새부적인 작업을 하기 위해
    pv_domain = pv_dwdv.get_super_structure(3,3)


    bss_dwdv = cp.Basis2D(pv_dwdv_sub)

    gen_vor = lambda x,y,c : plt.plot(x,y,color=c,marker='o',linestyle='None',markersize=15)
    gen_vor_A = lambda x,y: gen_vor(x,y,CLR_NODE[0,:])
    gen_vor_C = lambda x,y: gen_vor(x,y,CLR_NODE[1,:])
    gen_wall = lambda xx,yy,c: gen_directional_line(xx,yy,c=c,n=5,sharpness=1,biasness=0)
    gen_wall_r = lambda xx,yy: gen_wall(xx,yy,CLR_EDGE[0,:])
    gen_wall_g = lambda xx,yy: gen_wall(xx,yy,CLR_EDGE[2,:])
    gen_wall_b = lambda xx,yy: gen_wall(xx,yy,CLR_EDGE[1,:])

    p1=(2,1)# vorA
    p2=(1,2)# vorC
    p3=(1,-1)
    p4=(-1,1)

    bss_dwdv.add_artist(gen_wall_r,(p3,p1),label='wall_r')
    bss_dwdv.add_artist(gen_wall_g,(p2,p4),label='wall_g')
    bss_dwdv.add_artist(gen_wall_b,(p2,p1),label='wall_b')
    bss_dwdv.add_artist(gen_vor_A,(p1),label='vor_A')
    bss_dwdv.add_artist(gen_vor_C,(p2),label='vor_C')

    bss_domain = cp.Basis2D(pv_dwdv_sub) # to draw face
    gen_hex = lambda x,y,c: cp.Collection.Generator.gen_hexagon(x=x,y=y,c=c,phi=np.pi/2,r=1/3**.5)
    def gen_hex_all(x,y,cla_all):
        for i in range(3):
            for j in range(3):
                x_cntr,y_cntr = pv_dwdv.cal_xy_from_ij([i,j]).flatten()
                gen_hex_one = lambda x,y: gen_hex(x+x_cntr,y+y_cntr,cla_all[i,j,:])
                gen_hex_one(x,y)
    phase_3x3 = lambda x,y :gen_hex_all(x,y,CLR_FACE)
    bss_domain.add_artist(phase_3x3,(0,0),label='hexagons')

    xylim = ((-5,5),(-5,5))
    lp_dwdv = cp.LatticePoints2D(pv_dwdv) 
    lp_dwdv.generate_points_by_xylim(*xylim)
    lp_domain = cp.LatticePoints2D(pv_domain) 
    lp_domain.generate_points_by_xylim(*xylim)


    cry_dw_dv = cp.Crystal2D(bss_dwdv,lp_dwdv)
    cry_domain = cp.Crystal2D(bss_domain,lp_domain)
    plt.figure(figsize=(4,4))
    fig,ax = cry_dw_dv.plot_crystal()    
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    fullax(ax)
    savepng(fig,'dwn_toon')
    # plt.show()