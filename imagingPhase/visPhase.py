"""
visualize phasemap from phase info
"""

import numpy as np
import matplotlib.pyplot as plt
## Divide ##
def z3z3split(phi1,phi2):
  # phi1 from [0,2pi)
  # phi2 from [0,2pi)
  phi1 = phi1%(2*np.pi) # R -> [0,2pi)
  phi2 = phi2%(2*np.pi)
  phi1i = phi1/(2*np.pi) # [0,2pi) -> [0,1)
  phi2i = phi2/(2*np.pi)
  z3z31, rphi1 = divmod(phi1i, 1/3)
  z3z32, rphi2 = divmod(phi2i, 1/3)
  z3z3 = [z3z31,z3z32] # {0,1,2} #len2 list
  rphi12 = [rphi1*3,rphi2*3] # [0,1) #len2 list
  return z3z3,rphi12

def rlsplit(rphi12):
  is_close_10 = rphi12[0]>rphi12[1]#r-like|-> T, g-like|-> F
  if not is_close_10:
    rphi12 = [rphi12[1],rphi12[0]]
  rphi12 = np.array(rphi12)
  rphi12 = rphi12.reshape(2,1) #Basis B
  I_C__B = np.array([[1,-1],[0,1]]) #Identity
  rphi12_C = I_C__B@rphi12 #Basis C
  v_c1 = rphi12_C[0,0]
  v_c2 = rphi12_C[1,0]
  baryHHH_foo = np.array([1-v_c1-v_c2,v_c1,v_c2 ])
  baryHHH_foo = baryHHH_foo.reshape(3,1) #order of index: (00),(??),(11)
  return is_close_10,baryHHH_foo

def sixsplit(baryHHH_foo):
  p_BarHHH = np.sort(baryHHH_foo, axis=0)[::-1]
  z6 = np.argsort(baryHHH_foo, axis=0)[::-1]
  p_AffHHH = p_BarHHH[1:]
  I_AffHAC__AffHHH = np.array([[1,2],[1,-1]])
  p_AffHAC = I_AffHAC__AffHHH@p_AffHHH
  p_BarHAC = np.array([1-p_AffHAC[0]-p_AffHAC[1],p_AffHAC[0],p_AffHAC[1]])
  baryHAC_foo = p_BarHAC
  return z6,baryHAC_foo

## Conquer ##
def get_baryHHH(baryHHH_foo,is_close_10):
  baryHHH = np.zeros(4)
  baryHHH[0] = baryHHH_foo[0,0]
  baryHHH[3] = baryHHH_foo[2,0]
  # print('here',not(is_close_10)+1)
  baryHHH[(not(is_close_10))+1] = baryHHH_foo[1,0]
  return baryHHH #order of index: (00),(10),(01),(11)
def get_baryHAC(is_close_10,baryHAC_foo):
  # baryHAC = np.zeros(3)
  baryHAC = baryHAC_foo
  if not(is_close_10):
    baryHAC = baryHAC_foo[[0,2,1],[0]]
  return baryHAC #order of index: H,10L,01L
def phaseInfo(phi1,phi2):
  z3z3,rphi12 = z3z3split(phi1,phi2)
  is_close_10,baryHHH_foo = rlsplit(rphi12)
  z6,baryHAC_foo = sixsplit(baryHHH_foo)
  baryHHH = get_baryHHH(baryHHH_foo,is_close_10)
  baryHAC = get_baryHAC(is_close_10,baryHAC_foo)
  return baryHHH,baryHAC,z3z3,z6
def phaseInfo2(Phi1,Phi2):
  XY_shape = Phi1.shape
  BaryHHH = np.zeros((XY_shape[0],XY_shape[1],4))
  BaryHAC = np.zeros((XY_shape[0],XY_shape[1],3))
  Z3z3 = np.zeros((XY_shape[0],XY_shape[1],2))
  Z6 = np.zeros((XY_shape[0],XY_shape[1],3))
  for i in range(XY_shape[0]):
    for j in range(XY_shape[1]):
      phi1_now = Phi1[i,j]
      phi2_now = Phi2[i,j]

      baryHHH,baryHAC,z3z3,z6 = phaseInfo(phi1_now,phi2_now)

      BaryHHH[i,j,:] = baryHHH.flatten()
      BaryHAC[i,j,:] = baryHAC.flatten()
      Z3z3[i,j,0] = z3z3[0]
      Z3z3[i,j,1] = z3z3[1]
      Z6[i,j,0] = z6[0]
      Z6[i,j,1] = z6[1]
      Z6[i,j,2] = z6[2]
  return BaryHHH,BaryHAC,Z3z3,Z6


def vectorized_phaseInfo(Phi1, Phi2):
    """
    Vectorized version of phaseInfo2.
    Processes entire 2D arrays Phi1 and Phi2 without loops.
    """
    # 1. z3z3split 로직 벡터화
    phi1 = Phi1 % (2 * np.pi)
    phi2 = Phi2 % (2 * np.pi)
    phi1i = phi1 / (2 * np.pi)
    phi2i = phi2 / (2 * np.pi)

    # divmod를 floor division(//)과 modulo(%)로 대체
    z3z31 = phi1i // (1/3)
    rphi1 = phi1i % (1/3)
    z3z32 = phi2i // (1/3)
    rphi2 = phi2i % (1/3)

    # 결과를 마지막 축에 쌓아 (N, M, 2) 형태의 배열 생성
    Z3z3 = np.stack([z3z31, z3z32], axis=-1).astype(int)
    rphi12 = np.stack([rphi1 * 3, rphi2 * 3], axis=-1)

    # 2. rlsplit 로직 벡터화
    # is_close_10은 (N, M) 형태의 boolean 배열
    is_close_10 = rphi12[..., 0] > rphi12[..., 1]
    
    # if문을 np.where로 대체
    # is_close_10이 True이면 원래 순서, False이면 순서를 바꿈
    # is_close_10을 (N, M, 1)로 브로드캐스팅하여 rphi12와 연산
    swapped_rphi12 = rphi12[..., [1, 0]]
    rphi12_ordered = np.where(is_close_10[..., np.newaxis], rphi12, swapped_rphi12)

    # 행렬 곱셈을 위해 마지막 차원 추가 (N, M, 2) -> (N, M, 2, 1)
    rphi12_ordered = rphi12_ordered[..., np.newaxis]
    I_C__B = np.array([[1, -1], [0, 1]]) # (2, 2)
    # (2, 2) @ (N, M, 2, 1) -> (N, M, 2, 1)
    rphi12_C = I_C__B @ rphi12_ordered

    v_c1 = rphi12_C[..., 0, 0]
    v_c2 = rphi12_C[..., 1, 0]
    
    # 결과를 마지막 축에 쌓아 (N, M, 3) 형태의 배열 생성
    baryHHH_foo = np.stack([1 - v_c1 - v_c2, v_c1, v_c2], axis=-1)
    
    # 3. sixsplit 로직 벡터화
    # 정렬을 위해 (N, M, 3) -> (N, M, 3, 1) 형태로 변경
    baryHHH_foo_reshaped = baryHHH_foo[..., np.newaxis]
    
    # argsort와 sort를 axis=2 (3개 요소가 있는 축) 기준으로 수행
    Z6 = np.argsort(baryHHH_foo_reshaped, axis=2)[..., ::-1, 0].astype(int)
    p_BarHHH = np.sort(baryHHH_foo_reshaped, axis=2)[..., ::-1, :]
    
    p_AffHHH = p_BarHHH[..., 1:, :] # Shape: (N, M, 2, 1)
    I_AffHAC__AffHHH = np.array([[1, 2], [1, -1]]) # (2, 2)
    p_AffHAC = I_AffHAC__AffHHH @ p_AffHHH # Shape: (N, M, 2, 1)
    
    p_aff_0 = p_AffHAC[..., 0, 0]
    p_aff_1 = p_AffHAC[..., 1, 0]
    baryHAC_foo = np.stack([1 - p_aff_0 - p_aff_1, p_aff_0, p_aff_1], axis=-1)

    # 4. Conquer(get_bary*) 로직 벡터화
    # get_baryHHH
    BaryHHH = np.zeros(phi1.shape + (4,))
    BaryHHH[..., 0] = baryHHH_foo[..., 0]
    BaryHHH[..., 3] = baryHHH_foo[..., 2]
    # np.where를 사용하여 조건에 따라 값을 할당
    BaryHHH[..., 1] = np.where(is_close_10, baryHHH_foo[..., 1], 0)
    BaryHHH[..., 2] = np.where(~is_close_10, baryHHH_foo[..., 1], 0)

    # get_baryHAC
    swapped_baryHAC = baryHAC_foo[..., [0, 2, 1]]
    BaryHAC = np.where(is_close_10[..., np.newaxis], baryHAC_foo, swapped_baryHAC)

    return BaryHHH, BaryHAC, Z3z3, Z6

class phiPrinter():
    def __init__(self,phase):
        self.phase = phase
        print('phase uploaded')
        self.Info = vectorized_phaseInfo(self.Phi1, self.Phi2)
        print('phase calculated')
    @property
    def Phi1(self):      
      return self.phase[0]
    @property
    def Phi2(self):
      return - self.phase[1]
    @property
    def BaryHHH(self):
      return self.Info[0]
    @property
    def BaryHAC(self):
      return self.Info[1]
    @property
    def Z3z3(self):
      return self.Info[2]
    @property
    def Z6(self):
      return self.Info[3]
    def pHAC(self):
      fig = plt.figure(figsize=(20,20))
      HAC_argmax = np.argmax(self.BaryHAC, axis=2)
    #   fig = plt.figure(figsize=(20,20))
      ax = fig.imshow(HAC_argmax,cmap='gray')
      return ax
    
class DomainColoring:
  def __init__(self,Info,clrset=None):
    self.Info = Info
    self.z3z3hex = self._get_z3z3hex(Info)
    if clrset is None:
      self.clrset = self._set_defualt_clrset()
    else:
      self.clrset = clrset
  @property
  def phase9(self):
    indices_x = self.z3z3hex[:, :, 0].astype(int)
    indices_y = self.z3z3hex[:, :, 1].astype(int)
    phase9 = self.clrset[indices_x, indices_y]
    return phase9
  def show(self):
    plt.imshow(self.phase9)
    plt.axis('off')
    plt.show()
  
  @staticmethod
  def _get_z3z3hex(Info):
    HHH_argmax = np.argmax(Info[0], axis=2)
    mod01,mod10 = np.divmod(HHH_argmax ,2)
    hex1 = np.mod(Info[2][:,:,0] + mod10,3)
    hex2 = np.mod(Info[2][:,:,1] + mod01,3)
    z3z3hex = np.stack((hex1,hex2),axis=2)
    return z3z3hex
  @staticmethod
  def _set_defualt_clrset():
    colors = {
    "Red Bright": (1.0, 0.302, 0.302),   # #FF4D4D
    "Red Dark":   (0.478, 0.110, 0.110), # #7A1C1C
    "Green Bright": (0.302, 1.0, 0.302), # #4DFF4D
    "Green Dark":   (0.110, 0.478, 0.110), # #1C7A1C
    "Blue Bright":  (0.302, 0.58, 1.0),  # #4D94FF
    "Blue Dark":    (0.110, 0.110, 0.478) # #1C1C7A
    }
    p_00 = [.5,.5,.5]
    p_12 = [.0,.0,.0]
    p_21 = [1,1,1]

    p_10 = colors["Red Dark"]
    p_01 = colors["Green Dark"]
    p_22 = colors["Blue Dark"]

    p_20 = colors["Red Bright"]
    p_02 = colors["Green Bright"]    
    p_11 = colors["Blue Bright"]
    clrset = [[p_00,p_01,p_02],[p_10,p_11,p_12],[p_20,p_21,p_22]]
    clrset = np.array(clrset)
    return clrset
  
class DWallColoring:
  def __init__(self,Info,phase,dwThick=.2):
    self.Info = Info
    self.phase = phase
    self.dwThick = dwThick
        
  @property
  def rgb(self):
    Info = self.Info
    Z6 = Info[3]
    Is_close10 = Info[0][:,:,2]>0
    Phi1 = self.phase[0]
    Phi2 = - self.phase[1]
    dlt = self.dwThick

    classified3s = np.zeros(Z6.shape[0:2])
    # --- 상수 정의 ---
# 각 클래스에 해당하는 값들을 미리 정의합니다.
    blk = [1012, 1102, 120, 210]
    rlk = [1120, 1210, 12, 102]
    glk = [1021, 1201, 21, 201]
    class12 = np.array([blk, rlk, glk])  # Shape: (3, 4)

    # Z6의 마지막 차원을 숫자로 변환하기 위한 가중치입니다.
    weights = np.array([100, 10, 1])

    # --- 벡터화된 계산 ---
    # 가정: Z6 (H, W, 3) 형태의 3D 배열, Is_close10 (H, W) 형태의 2D 불리언 배열

    # 1. Z6의 각 [r, g, b] 값을 하나의 숫자로 변환합니다. (for 루프의 class6this 계산)
    #    (H, W, 3) @ (3,) -> (H, W) 형태의 배열이 됩니다.
    class6this_map = Z6 @ weights

    # 2. Is_close10 조건에 따라 1000을 더합니다. (for 루프의 class12this 계산)
    #    Is_close10이 True이면 1000, False이면 0이 더해집니다.
    class12this_map = class6this_map + 1000 * Is_close10

    # 3. class12 배열(3, 4)과 class12this_map(H, W)을 비교하여 각 픽셀이 어느 클래스에 속하는지 찾습니다.
    #    - class12this_map을 (H, W, 1, 1)로 확장합니다.
    #    - 브로드캐스팅을 통해 class12(3, 4)와 비교하여 (H, W, 3, 4) 형태의 불리언 배열을 생성합니다.
    #    - any(axis=3)를 통해 각 행(class)에 일치하는 값이 있는지 확인하여 (H, W, 3) 형태로 만듭니다.
    row_has_match = (class12this_map[..., np.newaxis, np.newaxis] == class12).any(axis=3)

    # 4. argmax(axis=2)를 사용하여 True 값을 가진 첫 번째 행(class)의 인덱스(0, 1, 또는 2)를 찾습니다.
    #    이것이 최종 결과인 classified3s가 됩니다.
    classified3s = np.argmax(row_has_match, axis=2)

    classified3s_rgb = np.stack((classified3s==1,classified3s==0,classified3s==2),axis=2)
    dwBW = Info[1][:,:,0]<dlt
    
    dw_rgb = classified3s_rgb * dwBW[:, :, np.newaxis]
    Z3z3,Rphi12 = z3z3split(Phi1,Phi2)
    coefs = np.array([[-2/4,4/4],[1/2,1/2],[4/4,-2/4]])

    chiral = np.einsum('ijk,li->jkl', np.array(Rphi12), coefs)
    chiral = chiral[:,:,[2,0,1]]
    chiral[:,:,2] = -chiral[:,:,2]
    chiral_ = np.mod(chiral+dlt/2,.5)-dlt/2
    chiral_ = -(1/dlt)*(chiral_-dlt/2)
    final = chiral_*dw_rgb

    from skimage.color import rgb2hsv,hsv2rgb #domain is white
    final2 = rgb2hsv(final)
    final2[:,:,1] = final2[:,:,2]
    final2[:,:,2] = 1
    rgb = hsv2rgb(final2)
    

    return rgb

  def show(self):
    plt.imshow(self.rgb)
    plt.axis('off')
    # plt.show()

class DVertexColoring:
  def __init__(self,Info):
    self.Info = Info
  @property
  def baryHAC(self):
    return self.Info[1]
  @property
  def rgb(self):
    baryHAC = self.baryHAC
    rgb = self.apply_ternary_colormap(baryHAC)
    return rgb
  def show(self):
    plt.imshow(self.rgb)
    plt.axis('off')
  @staticmethod
  def apply_ternary_colormap(
    data_array,
    color1=(1, 1, 1),  # v1=1에 해당하는 색상 (기본값: Red)
    color2=(0, 1, 0),  # v2=1에 해당하는 색상 (기본값: Green)
    color3=(1, 0, 1)   # v3=1에 해당하는 색상 (기본값: Blue)
):
    """
    (H, W, 3) 형태의 바리센트릭 좌표 배열에 터너리 컬러맵을 적용합니다.

    Args:
        data_array (np.ndarray): (H, W, 3) 형태의 입력 배열.
                                 마지막 축은 (v1, v2, v3) 값을 가집니다.
        color1 (tuple): v1에 해당하는 RGB 색상.
        color2 (tuple): v2에 해당하는 RGB 색상.
        color3 (tuple): v3에 해당하는 RGB 색상.

    Returns:
        np.ndarray: (H, W, 3) 형태의 RGB 색상 배열.
    """
    # 입력 데이터에서 v1, v2, v3 좌표를 분리합니다.
    v1 = data_array[..., 0]
    v2 = data_array[..., 1]
    v3 = data_array[..., 2]

    # 입력된 색상 핸들을 numpy 배열로 변환합니다.
    c1 = np.array(color1)
    c2 = np.array(color2)
    c3 = np.array(color3)

    # 브로드캐스팅을 위해 v1,v2,v3의 차원을 (H,W) -> (H,W,1)로 확장하고
    # 각 색상과 곱한 뒤 모두 더하여 최종 RGB 값을 계산합니다.
    rgb_array = (v1[..., np.newaxis] * c1 +
                 v2[..., np.newaxis] * c2 +
                 v3[..., np.newaxis] * c3)

    # 부동소수점 오차 등으로 인해 값이 0~1 범위를 벗어날 경우를 대비해 clip 처리합니다.
    return np.clip(rgb_array, 0, 1)
class PhaseMapVisualizer:
  pass    

      
    
