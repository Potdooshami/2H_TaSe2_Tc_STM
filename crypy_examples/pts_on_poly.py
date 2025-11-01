import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

def lattice_points_in_polygon(a1, a2, polygon_vertices):
    a1 = np.array(a1, dtype=float)
    a2 = np.array(a2, dtype=float)
    poly = Polygon(polygon_vertices)
    
    minx, miny, maxx, maxy = poly.bounds
    
    # (x,y) -> (i,j) 변환 행렬
    A = np.column_stack((a1, a2))
    Ainv = np.linalg.inv(A)
    
    corners = np.array([
        [minx, miny],
        [maxx, maxy],
        [minx, maxy],
        [maxx, miny]
    ])
    ij_corners = (Ainv @ corners.T).T
    i_min, j_min = np.floor(ij_corners.min(axis=0)).astype(int)
    i_max, j_max = np.ceil(ij_corners.max(axis=0)).astype(int)
    
    lattice_points = []
    lattice_indices = []
    for i in range(i_min-1, i_max+2):
        for j in range(j_min-1, j_max+2):
            p = i * a1 + j * a2
            if poly.contains(Point(p)):
                lattice_points.append(tuple(p))
                lattice_indices.append((i, j))
    

    return (lattice_points,lattice_indices)


def plot_lattice_points(a1, a2, polygon_vertices):
    points,indices = lattice_points_in_polygon(a1, a2, polygon_vertices)
    poly = np.array(polygon_vertices )  # 닫힌 경계선용
    poly = np.vstack((poly, poly[0]))  # 닫힌 경계선용
    # 기본 격자 시각화용
    a1 = np.array(a1)
    a2 = np.array(a2)
    
    fig, ax = plt.subplots(figsize=(6,6))
    
    # polygon 그리기
    ax.plot(poly[:,0], poly[:,1], 'k-', lw=1.5, label="Polygon")
    
    # lattice points 표시
    if points:
        pts = np.array(points)
        ax.scatter(pts[:,0], pts[:,1], color='red', s=25, label='Lattice points')
    
    # basis vector 시각화
    origin = np.array([0,0])
    ax.arrow(*origin, *a1, color='blue', head_width=0.1, length_includes_head=True)
    ax.arrow(*origin, *a2, color='green', head_width=0.1, length_includes_head=True)
    ax.text(*(a1*1.1), "a1", color='blue')
    ax.text(*(a2*1.1), "a2", color='green')
    
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("Lattice points inside polygon")
    plt.show()

if __name__ == "__main__":
    R = 15
    R = R +0.01 

    tht = 2*np.pi*(1/12+np.arange(0,6)/6)
    v_xy =R*(2/np.sqrt(3))*np.array([np.cos(tht),np.sin(tht)])
    print(v_xy.shape)
    print(v_xy)

    a1 = [1.0, 0.0]
    a2 = [-0.5, np.sqrt(3)/2]
    polygon = v_xy.transpose()

    plot_lattice_points(a1, a2, polygon)
    plot_lattice_points(a1, a2, polygon)