import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Voronoi

class TrackedInstance:
    _all_instances = {}
    def __init__(self):
        cls = type(self)
        if cls not in TrackedInstance._all_instances:
            TrackedInstance._all_instances[cls] = []
        TrackedInstance._all_instances[cls].append(self)
    @classmethod
    def get_instances_df(cls):
        instances_list = TrackedInstance._all_instances.get(cls, [])
        if not instances_list:
            print(f"No instances found for class {cls.__name__}.")
            return pd.DataFrame()
        data = [vars(inst) for inst in instances_list]
        return pd.DataFrame(data)
    @staticmethod
    def print_all_instances_class_wise():
        print("All instances class-wise:")
        if not TrackedInstance._all_instances:
            print("No instances found.")
            print("_"*40)
            return
        for cls, instances_list in TrackedInstance._all_instances.items():
            print(f"Class: {cls.__name__}, Number of instances: {len(instances_list)}")
            df = cls.get_instances_df()
            print(df.to_string())
        print("\n" + "-" * 40)
        
    @classmethod
    def clear_instances(cls):
        if cls in TrackedInstance._all_instances:
            TrackedInstance._all_instances.pop(cls)
            print(f" '{cls.__name__}' class instances cleared.")
        else:
            print(f"No instances found for class '{cls.__name__}'.")
    @staticmethod
    def clear_all_project_instances():
        TrackedInstance._all_instances.clear()
        print("All class instances cleared.")
                
class CrystalGenerator2D:
    def __init__(self,a1,a2,O=(0,0)):
        self.minimal_unit_vector = (a1,a2)
        self.unit_vectors = [PrimitiveVector2D(a1,a2,O=O)]
    def add_superstructure(self,n1,n2):
        self.unit_vectors.append(self.unit_vectors[0].get_super_structure(n1,n2))
class PrimitiveVector2D(TrackedInstance):
    def __init__(self,a1,a2,O=np.array((0,0)),gizmowidth=10):
        self.a1 = np.array(a1).reshape(2)
        self.a2 = np.array(a2).reshape(2)
        self.O = np.array(O).reshape(2)
        self.gizmowidth = gizmowidth
        self.clr_O = [0,0,0]
        self.clr_a1 = [1,0,0]
        self.clr_a2 = [0,1,0]
        super().__init__()
    @property
    def I_xy__a12(self):
        return np.array([self.a1,self.a2]).T
    @property
    def clrs(self):
        return np.array([self.clr_O,self.clr_a1,self.clr_a2])
    def cal_xy_from_ij(self,ij):
        ij = np.array(ij).reshape(-1,2)
        return self.O + ij[:,0].reshape(-1,1)*self.a1 + ij[:,1].reshape(-1,1)*self.a2
    def cal_ij_from_xy(self,xy):
        xy = np.array(xy).reshape(-1,2)
        A = self.I_xy__a12
        if np.abs(np.linalg.det(A))<1e-9:
            raise ValueError("a1 and a2 are linearly dependent")
        A_inv = np.linalg.inv(A)
        ij = A_inv@(xy - self.O).T
        return ij.T
    def plot_gizmo(self):
        for ind_a12 in range(2):
            dx,dy =self.I_xy__a12[:,ind_a12]
            plt.arrow(self.O[0],self.O[1],dx,dy,
                      head_width=0.1,head_length=0.2,color=self.clrs[ind_a12+1,:],length_includes_head=True,
                      linewidth=self.gizmowidth)
        plt.plot(self.O[0],self.O[1],'o',color=self.clr_O,markersize=self.gizmowidth*2)
        plt.axis('equal')
    def plot_wigner_seitz_2d(self):
        vertices = self.get_wigner_seitz_vertices_2d()
        vertices = vertices.T + self.O.reshape(2,1)        
        fill_wigner =plt.fill(vertices[0,:],vertices[1,:],edgecolor='k',fill=False,linewidth=self.gizmowidth/2)
        plt.axis('equal')
        return fill_wigner
    def plot_paral_2d(self):
        vertices = self.I_xy__a12@np.array([[0,1,1,0],[0,0,1,1]]) + self.O.reshape(2,1)
        fill_wigner =plt.fill(vertices[0,:],vertices[1,:],edgecolor='k',fill=False,linewidth=self.gizmowidth/2)
        plt.axis('equal')
        return fill_wigner
    def plot_all(self):
        self.plot_gizmo()
        self.plot_wigner_seitz_2d()
        self.plot_paral_2d()    
    def get_super_structure(self,n1,n2):
        return PrimitiveVector2D(n1*self.a1,n2*self.a2,O=self.O,gizmowidth=self.gizmowidth/(n1+n2))
    def get_sub_structure(self,n1,n2):
        return self.get_super_structure(1/n1,1/n2)
    def get_wigner_seitz_vertices_2d(self) -> np.ndarray:
        """
        주어진 2D 기저 벡터에 대한 비그너-자이츠 셀의 꼭짓점 좌표를 계산합니다.

        Args:
            basis_vectors (np.ndarray): 각 행이 기저 벡터인 2x2 numpy 배열.
                                        예: np.array([[1, 0], [0, 1]])

        Returns:
            np.ndarray: 비그너-자이츠 셀을 구성하는 꼭짓점들의 (x, y) 좌표가
                        담긴 (N, 2) 형태의 numpy 배열. N은 꼭짓점의 개수입니다.
        """
        basis_vectors = self.I_xy__a12.T
        if not isinstance(basis_vectors, np.ndarray) or basis_vectors.shape != (2, 2):
            raise TypeError("basis_vectors는 2x2 형태의 numpy 배열이어야 합니다.")

        a1 = self.a1#basis_vectors[0]
        a2 = self.a2#basis_vectors[1]

        # --- 1. 원점 주변의 격자점 생성 ---
        # 비그너-자이츠 셀은 가장 가까운 이웃에 의해 결정되므로,
        # 보통 원점 주변 5x5 격자점이면 충분합니다.
        lattice_points = np.array([n * a1 + m * a2 
                                for n in range(-2, 3) 
                                for m in range(-2, 3)])

        # --- 2. 보로노이 다이어그램 계산 ---
        vor = Voronoi(lattice_points)

        # --- 3. 원점에 해당하는 셀의 꼭짓점 찾기 ---
        # 격자점 배열에서 원점(0,0)의 인덱스를 찾습니다.
        # 부동소수점 오차를 고려하여 np.isclose 사용
        origin_idx = np.where(np.all(np.isclose(lattice_points, [0, 0]), axis=1))[0][0]
        
        # 원점에 해당하는 보로노이 영역(region)의 인덱스를 가져옵니다.
        region_idx = vor.point_region[origin_idx]
        
        # 해당 영역을 구성하는 꼭짓점(vertex)들의 인덱스를 가져옵니다.
        vertex_indices = vor.regions[region_idx]
        
        # -1은 무한대를 의미하므로 유효한 꼭짓점만 필터링합니다.
        if -1 in vertex_indices:            
            # 일반적으로 가장자리 셀이 아니면 발생하지 않습니다.
            # 혹시 모를 경우를 대비해 유효한 인덱스만 사용합니다.
            valid_indices = [i for i in vertex_indices if i != -1]
            return vor.vertices[valid_indices]
        return vor.vertices[vertex_indices]
                      
class LatticePoints2D(TrackedInstance):
    def __init__(self,primitive_vector:PrimitiveVector2D):
        self.primitive_vector = primitive_vector
        super().__init__()
    def generate_points_by_range(self,n1_range,n2_range):    
        self.Indices = np.meshgrid(np.arange(n1_range[0],n1_range[1]+1),np.arange(n2_range[0],n2_range[1]+1),indexing='ij')
        self.Indices = np.array(self.Indices).reshape(2,-1).T
    def generate_points_by_xylim(self,xrng,yrng):
        xmin, xmax = xrng
        ymin, ymax = yrng
        a1 = self.primitive_vector.a1
        a2 = self.primitive_vector.a2
        O = self.primitive_vector.O
        self.Indices = self.find_lattice_indices_in_rect(a1, a2, O, xmin, xmax, ymin, ymax)
    def generate_points_by_manual(self,ijList):
        self.Indices = np.array(ijList).reshape(-1,2)
    
    @property
    def xy(self):
        return self.primitive_vector.cal_xy_from_ij(self.Indices)
    def plot_scatter(self,**kwargs):
        xy = self.xy
        plt.scatter(xy[:,0],xy[:,1],**kwargs)
        plt.axis('equal')
    def plot_text(self,**kwargs):
        xy = self.xy
        str_list = [f'({row[0]},{row[1]})' for row in self.Indices]
        for x,y,s in zip(xy[:,0],xy[:,1],str_list):
            plt.text(x,y,s,**kwargs)        
        plt.axis('equal')
    def plot_line(self,**kwargs):
        xy = self.xy
        plt.plot(xy[:,0],xy[:,1],**kwargs)
        plt.axis('equal')

    @staticmethod
    def find_lattice_indices_in_rect(a1, a2, O, xmin, xmax, ymin, ymax):
        """
        주어진 2D 격자에서 특정 직사각형 범위 내에 있는 모든 격자점의 인덱스를 찾습니다.

        Args:
            a1 (tuple or np.ndarray): 첫 번째 기저 벡터 (ax1, ay1).
            a2 (tuple or np.ndarray): 두 번째 기저 벡터 (ax2, ay2).
            O (tuple or np.ndarray): 격자의 원점 (Ox, Oy).
            xmin, xmax, ymin, ymax (float): 직사각형 경계.

        Returns:
            np.ndarray: 직사각형 내에 있는 모든 격자점의 (i, j) 인덱스가 담긴
                        (N, 2) 형태의 numpy 배열. N은 점의 개수입니다.
        """
        # 1. 벡터와 행렬 설정
        a1 = np.array(a1, dtype=float)
        a2 = np.array(a2, dtype=float)
        O = np.array(O, dtype=float)
        
        # 변환 행렬 M = [a1, a2] (a1, a2를 열벡터로 가짐)
        M = np.stack([a1, a2], axis=1)
        
        # M이 역행렬을 가질 수 있는지 확인 (a1, a2가 평행하지 않은지)
        if np.abs(np.linalg.det(M)) < 1e-9:
            raise ValueError("기저 벡터 a1, a2가 서로 평행하여 2D 격자를 만들 수 없습니다.")
        
        # 역변환 행렬
        M_inv = np.linalg.inv(M)

        # 2. 직사각형의 네 꼭짓점을 정의
        corners_xy = np.array([
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax]
        ])

        # 3. 네 꼭짓점을 격자 좌표계(i, j)로 변환
        # (corner - O)를 계산해야 하므로 O를 각 corner에 맞게 브로드캐스팅
        corners_ij = M_inv @ (corners_xy - O).T
        
        # 4. 변환된 좌표들로부터 i와 j의 탐색 범위를 결정
        i_min, j_min = np.floor(corners_ij.min(axis=1))
        i_max, j_max = np.ceil(corners_ij.max(axis=1))
        
        # 5. 해당 범위 내의 모든 (i, j) 정수 쌍을 생성하고 검사
        valid_indices = []
        for i in range(int(i_min), int(i_max) + 1):
            for j in range(int(j_min), int(j_max) + 1):
                # (i, j)에 해당하는 실제 (x, y) 좌표 계산
                point_xy = O + i * a1 + j * a2
                
                # 이 점이 실제로 직사각형 안에 있는지 최종 확인
                if (xmin <= point_xy[0] <= xmax) and (ymin <= point_xy[1] <= ymax):
                    valid_indices.append([i, j])
                    
        return np.array(valid_indices)
class Basis2D(TrackedInstance):
    def __init__(self,primitive_vector:PrimitiveVector2D):
        self.primitive_vector = primitive_vector
        self._artist_list = []
        super().__init__()
    def add_artist(self,generator,v_a12,label):
        self._artist_list.append({'generator':generator,'v_a12':v_a12,'label':label})
    @property
    def basis_df(self):
        return pd.DataFrame(self._artist_list) 
    def plot_basis(self):
        basis_df = self.basis_df
        for idx,row in basis_df.iterrows():
            v_a12 = np.array(row['v_a12']).reshape(-1,2)
            v_xy = self.primitive_vector.cal_xy_from_ij(v_a12)
            row['generator'](v_xy[:,0],v_xy[:,1])
            plt.text(v_xy[:,0].mean(),v_xy[:,1].mean(),row['label'],color='k')
            plt.axis('equal')

class Crystal2D(TrackedInstance):
    def __init__(self,basis:Basis2D,lattice:LatticePoints2D):
        self._basis = basis 
        self._lattice = lattice
        super().__init__()
    def plot_crystal(self,x_=0,y_=0):
        basis_df = self._basis.basis_df
        Indices = self._lattice.Indices
        for idx,row in basis_df.iterrows():
            v_a12 = np.array(row['v_a12']).reshape(-1,2)#sub unit cell position
            v_xy = self._basis.primitive_vector.cal_xy_from_ij(v_a12)
            for ij in Indices:
                O_ij = self._lattice.primitive_vector.cal_xy_from_ij(ij)
                O_ij = O_ij.flatten()
                x = v_xy[:,0]+O_ij[0] + x_
                y = v_xy[:,1]+O_ij[1] + y_
                row['generator'](x,y)
        f = plt.gcf()
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')    
        return (f,ax)

class Collection:
    class Generator:
        @staticmethod
        def gen_regular_polygon(n,x=0,y=0,r=1,c='blue',phi=0,**kwargs):
            angles = np.linspace(0, 2 * np.pi, n+1)[:-1] + phi
            x_vertices = x + r * np.cos(angles)
            y_vertices = y + r * np.sin(angles)
            return plt.fill(x_vertices, y_vertices, color=c, alpha=1,**kwargs)
        @staticmethod
        def gen_hexagon(**kwargs):
            return Collection.Generator.gen_regular_polygon(6,**kwargs)
            






if __name__ == "__main__":
    import crypy as cp
    import matplotlib.pyplot as plt
    import importlib
    importlib.reload(cp)
    a1 = [1,0]
    a2 = [-0.5,3**0.5/2]
    pv = cp.PrimitiveVector2D(a1,a2)
    pv2 = pv.get_sub_structure(3,3)
    bss1 = cp.Basis2D(pv2)
    
    
    atomgen1 = lambda x,y: plt.plot(x,y,color='r',marker='o',linestyle='None',markersize=10)
    atomgen2 = lambda x,y: plt.plot(x,y,color='g',marker='o',linestyle='None',markersize=10)
    bondgen = lambda xx,yy: plt.plot(xx,yy,color='b')

    p1=(2,1)
    p2=(1,2)
    p3=(1,-1)
    p4=(-1,1)
    
    bss1.add_artist(bondgen,(p1,p3),label='bond1')
    bss1.add_artist(bondgen,(p2,p4),label='bond2')
    bss1.add_artist(bondgen,(p1,p2),label='bond3')
    bss1.add_artist(atomgen1,(p1),label='atom1')
    bss1.add_artist(atomgen2,(p2),label='atom2')
    

    lp = cp.LatticePoints2D(pv) 
    lp.generate_points_by_xylim((-5,5),(-5,5))
    

    cry = cp.Crystal2D(bss1,lp)
    f,ax =cry.plot_crystal()    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)    
    plt.show()



