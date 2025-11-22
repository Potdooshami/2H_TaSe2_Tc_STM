from imagingPhase import visPhase as vp
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage import distance_transform_edt
image_paths = ['solLatt_shiftmap_noCrop.png',
'iccdw_kmap_ik0_idt0.png',
'iccdw_kmap_ik1_idt0.png',
'iccdw_kmap_ik2_idt0.png',
'topological_encoding/segment9.png']
def label_by_size(bw):
    from skimage.measure import label
    bw_labeled = label(bw)
    lblList = np.unique(bw_labeled)
    foo = bw_labeled[:,:,np.newaxis] == lblList.reshape(1,1,-1)
    argsort = foo.sum(axis=0).sum(axis=0).argsort()
    bw_stack = foo[:,:,argsort[::-1]]
    return bw_stack
def big_one(bw):
    bw_stack = label_by_size(bw)
    return bw_stack[:,:,1]
def fill_gaps_nearest(label_map):
    # 0이 아닌 부분(유효 영역)에 대한 인덱스를 구함
    # distance_transform_edt의 return_indices=True 등을 응용하거나
    # 아래와 같이 nearest interpolation을 수행할 수 있습니다.
    
    # 1. 유효 영역과 빈 영역 구분 (0이면 True)
    mask = (label_map == 0)
    
    # 2. 가장 가까운 유효 픽셀까지의 거리와 인덱스 계산
    # indices[0]은 y좌표, indices[1]은 x좌표의 매핑
    distances, indices = distance_transform_edt(mask, return_indices=True)
    
    # 3. 빈 픽셀을 가장 가까운 유효 픽셀의 값으로 대치
    filled_map = label_map[tuple(indices)]
    
    return filled_map





# imgs = list(map(mpimg.imread, image_paths))
cset = vp.DomainColoring._set_defualt_clrset()
img = mpimg.imread('topological_encoding/segment9.png')
img = img[:,:,0:3]
bws = []
for ii in range(3):
    for jj in range(3):
        clr = cset[ii,jj].reshape(1,1,3)
        get_dist = lambda img,clr :np.sum(np.abs(img-clr),axis=-1)
        dist = get_dist(img,clr)
        bw = dist<.01
        bws.append(bw)


bw_split = []       

bw = bws[6]
bw_stack =  label_by_size(bw)
bw_split.append(bw_stack[:,:,1])
bw_split.append(bw_stack[:,:,2])
bw = bws[7]
bw_stack =  label_by_size(bw)
bw_split.append(bw_stack[:,:,1])
bw_split.append(bw_stack[:,:,2]+bw_stack[:,:,3])
bws_full = bws[0:6]
bws_full.append(bws[8])
bws_full = list(map(big_one,bws_full))
bws_full = bws_full+bw_split
bws_full = np.stack(bws_full,axis=-1)
lbl3 =np.arange(1,12).reshape(1,1,11)
lbl_full = (bws_full)*lbl3
lbl_full =lbl_full.sum(axis=-1) 

lbl_fin = fill_gaps_nearest(lbl_full)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.imshow(lbl_fin)
    plt.show()
