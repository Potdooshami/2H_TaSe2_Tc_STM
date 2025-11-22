from topological_encoding.segmentation import lbl_fin as lbl
import numpy as np
from skimage import segmentation, measure
from scipy.spatial import KDTree
from matplotlib import pyplot as plt

def extract_region_boundaries_and_junctions(label):
    H, W = label.shape
    
    # 1) Boundary mask (True = boundary pixel)
    boundary_mask = segmentation.find_boundaries(label, mode='outer')

    # 2) Extract polylines (each is Nx2 array: [ [y,x], ... ]).
    #   0.5 등치선(contour)으로 경계를 polygonize.
    contours = measure.find_contours(boundary_mask.astype(float), 0.5)

    boundary_edges = []
    for cnt in contours:
        # cnt: [[y,x], ...] float coords
        Ys = cnt[:, 0]
        Xs = cnt[:, 1]
        boundary_edges.append((Xs, Ys))

    # 3) Junction detection
    #    아이디어: 모든 polyline의 끝점들(endpoints)을 모아
    #             가까운 것끼리 cluster
    endpoints = []
    for Xs, Ys in boundary_edges:
        endpoints.append([Xs[0], Ys[0]])
        endpoints.append([Xs[-1], Ys[-1]])

    endpoints = np.array(endpoints)
    
    # cluster endpoints (merge points closer than eps)
    eps = 1.5   # distance threshold (pixel units)
    tree = KDTree(endpoints)
    
    used = np.zeros(len(endpoints), dtype=bool)
    junctions = []

    for i in range(len(endpoints)):
        if used[i]: continue
        idx = tree.query_ball_point(endpoints[i], eps)
        pts = endpoints[idx]
        center = pts.mean(axis=0)
        junctions.append(center)
        used[idx] = True

    return boundary_edges, junctions

import numpy as np
from collections import defaultdict
from skimage import measure
from scipy.spatial import KDTree

def pairwise_region_edges_and_junctions(label,
                                        background_label=None,
                                        neighbor_mode='4',   # '4' or '8'
                                        contour_level=0.5,
                                        endpoint_merge_eps=1.5,
                                        min_contour_length=3):
    """
    Input:
      label: 2D ndarray of integer labels (shape HxW)
      background_label: if not None, ignore boundaries where one side == background_label
      neighbor_mode: '4' or '8' connectivity when checking neighboring pixels
      contour_level: level passed to find_contours (0.5 typical for binary masks)
      endpoint_merge_eps: distance threshold to merge nearby endpoints into junctions (pixels)
      min_contour_length: drop very short contours (in points)
    Returns:
      edges_by_pair: dict mapping frozenset({a,b}) -> list of polylines,
                     each polyline is an (Xs, Ys) tuple of numpy arrays (x coords, y coords)
      junctions: list of dicts: {"xy": (x,y), "touching_pairs": set([...]), "touching_labels": set([...])}
    """
    H, W = label.shape
    # offsets for 4- or 8-neighbors
    if neighbor_mode == '4':
        offsets = [(-1,0),(1,0),(0,-1),(0,1)]
    else:
        offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    # Step 1: For each pixel, record all neighbor pairs (min,max) where label differs
    pair_pixel_mask = defaultdict(lambda: np.zeros_like(label, dtype=np.uint8))
    for dy,dx in offsets:
        # shift the label array
        shifted = np.full_like(label, fill_value=-9999)
        if dy >= 0:
            y_src = slice(0, H-dy)
            y_dst = slice(dy, H)
        else:
            y_src = slice(-dy, H)
            y_dst = slice(0, H+dy)
        if dx >= 0:
            x_src = slice(0, W-dx)
            x_dst = slice(dx, W)
        else:
            x_src = slice(-dx, W)
            x_dst = slice(0, W+dx)

        shifted[y_dst, x_dst] = label[y_src, x_src]
        # valid positions where shifted != sentinel
        valid = (shifted != -9999)
        diff = (label != shifted) & valid

        ys, xs = np.nonzero(diff)
        for y,x in zip(ys, xs):
            a = int(label[y,x])
            b = int(shifted[y,x])
            # optionally ignore background-border edges
            if background_label is not None and (a == background_label or b == background_label):
                continue
            # create canonical pair key
            key = tuple(sorted((a,b)))
            pair_pixel_mask[key][y,x] = 1

    # Step 2: For each pair mask, extract contours (polylines)
    edges_by_pair = {}
    for key, mask in pair_pixel_mask.items():
        if mask.sum() == 0:
            continue
        # find_contours expects float image
        contours = measure.find_contours(mask.astype(float), contour_level)
        polylines = []
        for cnt in contours:
            if cnt.shape[0] < min_contour_length:
                continue
            Ys = cnt[:, 0]   # row coords (y)
            Xs = cnt[:, 1]   # col coords (x)
            polylines.append((Xs.copy(), Ys.copy()))
        if polylines:
            edges_by_pair[frozenset(key)] = polylines

    # Step 3: Junction detection
    # 3a) endpoints clustering from polylines
    endpoints = []
    endpoint_info = []  # (pair_key, polyline_index, is_start_bool)
    for pair, polys in edges_by_pair.items():
        for pi, (Xs, Ys) in enumerate(polys):
            start = (Xs[0], Ys[0])
            end   = (Xs[-1], Ys[-1])
            endpoints.append(start)
            endpoint_info.append((pair, pi, True))
            endpoints.append(end)
            endpoint_info.append((pair, pi, False))

    junctions = []
    if endpoints:
        pts = np.array(endpoints)  # shape (M,2) with cols (x,y)
        tree = KDTree(pts)
        used = np.zeros(len(pts), dtype=bool)
        for i in range(len(pts)):
            if used[i]:
                continue
            idxs = tree.query_ball_point(pts[i], endpoint_merge_eps)
            used[idxs] = True
            cluster_pts = pts[idxs]
            center = cluster_pts.mean(axis=0)
            # collect touching labels/pairs
            touching_pairs = set()
            touching_labels = set()
            for j in idxs:
                pair_key, poly_idx, is_start = endpoint_info[j]
                touching_pairs.add(tuple(sorted(pair_key)))
                for lab in pair_key:
                    touching_labels.add(lab)
            junctions.append({"xy": (float(center[0]), float(center[1])),
                              "touching_pairs": touching_pairs,
                              "touching_labels": touching_labels,
                              "num_endpoints": len(idxs)})

    # 3b) (optional) detect pixels where >=3 distinct labels touch in 3x3 neighborhood
    multi_label_pixels = []
    pad = 1
    padded = np.pad(label, pad_width=pad, mode='constant', constant_values=-9999)
    for y in range(pad, pad+H):
        for x in range(pad, pad+W):
            neighborhood = padded[y-1:y+2, x-1:x+2]
            unique = set(np.unique(neighborhood))
            unique.discard(-9999)
            if background_label is not None:
                unique.discard(background_label)
            if len(unique) >= 3:
                multi_label_pixels.append((x-pad, y-pad, tuple(sorted(unique))))
    # convert multi_label pixels to distinct junctions (cluster by proximity)
    if multi_label_pixels:
        pts_ml = np.array([[mx, my] for (mx,my,_) in multi_label_pixels])
        tree_ml = KDTree(pts_ml)
        used_ml = np.zeros(len(pts_ml), dtype=bool)
        for i in range(len(pts_ml)):
            if used_ml[i]: continue
            idxs = tree_ml.query_ball_point(pts_ml[i], 1.5)
            used_ml[idxs] = True
            cluster = pts_ml[idxs]
            center = cluster.mean(axis=0)
            labels_touch = set()
            for j in idxs:
                labels_touch.update(multi_label_pixels[j][2])
            junctions.append({"xy": (float(center[0]), float(center[1])),
                              "touching_pairs": None,
                              "touching_labels": labels_touch,
                              "num_endpoints": 0,
                              "multi_label_pixel_cluster": True})

    return edges_by_pair, junctions




edges_by_pair, junctions = pairwise_region_edges_and_junctions(lbl)

import pandas as pd

def edges_by_pair_to_edge_dataframe(edges_by_pair):
    rows = []

    for pair, edge_list in edges_by_pair.items():
        r1, r2 = tuple(pair)

        for (Xs, Ys) in edge_list:
            rows.append({
                "R1": r1,
                "R2": r2,
                "X": list(map(float, Xs)),
                "Y": list(map(float, Ys))
            })

    df = pd.DataFrame(rows, columns=["R1", "R2", "X", "Y"])
    return df
df = edges_by_pair_to_edge_dataframe(edges_by_pair)
print(len(df))
print(df.shape)
def symmetric_average(xs):
    n = len(xs)
    result = []
    left = 0
    right = n - 1

    while left < right:
        result.append((xs[left] + xs[right]) / 2)
        left += 1
        right -= 1

    # 홀수 길이면 중앙값을 그대로 append
    if left == right:
        result.append(xs[left])

    return result

import numpy as np

def circular_shift_max_half_distance(X, Y):
    """
    X, Y : 동일 길이의 리스트 (경계 polyline 혹은 boundary edge)
    출력 : X_shifted, Y_shifted
    조건 : p0와 p_(n//2) 사이의 거리가 최대가 되는 shift 선택
    """

    X = np.array(X)
    Y = np.array(Y)
    n = len(X)

    # 안전성 체크
    if n < 2:
        return X, Y

    best_shift = 0
    best_dist = -1.0

    for shift in range(n):
        # shift 적용 후의 index
        mid_idx = (shift + n//2) % n

        # 거리 계산
        dx = X[shift] - X[mid_idx]
        dy = Y[shift] - Y[mid_idx]
        dist = dx*dx + dy*dy   # sqrt 불필요 (비교만 하면 됨)

        # 최대 거리 갱신
        if dist > best_dist:
            best_dist = dist
            best_shift = shift

    # 최종 best shift 적용
    X_shifted = np.roll(X, -best_shift)
    Y_shifted = np.roll(Y, -best_shift)

    return X_shifted.tolist(), Y_shifted.tolist()


# ---------------------
# 사용 예시
# ---------------------
X = [0, 1, 3, 6, 7, 5]
Y = [0, 2, 4, 4, 1, 0]

Xs, Ys = circular_shift_max_half_distance(X, Y)



x =df.iloc[ind]['X']
y = df.iloc[ind]['Y']
x,y = circular_shift_max_half_distance(x,y)
x = symmetric_average(x)
y = symmetric_average(y)

breakpoint()
if __name__ == '__main__':
    plt.imshow(lbl)
    for ind in range(21):# x = x[0:len(x)//2]        # y = y[0:len(y)//2]
        plt.plot(x,y)
        H, W = lbl.shape
        plt.xlim(0,H)
        plt.ylim(W,0)
    plt.show()
# print(edges_by_pair)
# print(boundary_edges[0])
# for ind in range(5):
#     plt.plot(edges_by_pair[ind][0],edges_by_pair[ind][1])
# plt.show()
# print(junctions)
