import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from scipy.spatial import Delaunay, Voronoi, ConvexHull


__all__ = ["GeometryCore", "RenderQueue", "DualMesh"]


class GeometryCore:
    """
    Pure geometric and mathematical utilities isolated from rendering and plotting.
    """

    @staticmethod
    def liang_barsky_clip(p1, p2, box=[0.0, 1.0, 0.0, 1.0]):
        """
        Clips a 2D line segment to a bounding box using Liang-Barsky algorithm.
        Returns clipped (start, end) points, or None if completely outside.
        """
        x0, y0 = p1
        x1, y1 = p2
        dx, dy = x1 - x0, y1 - y0

        p = [-dx, dx, -dy, dy]
        q = [x0 - box[0], box[1] - x0, y0 - box[2], box[3] - y0]

        u1, u2 = 0.0, 1.0

        for i in range(4):
            if p[i] == 0:
                if q[i] < 0:
                    return None
            else:
                t = q[i] / p[i]
                if p[i] < 0:
                    u1 = max(u1, t)
                else:
                    u2 = min(u2, t)

        if u1 > u2:
            return None

        clipped_p1 = np.array([x0 + u1 * dx, y0 + u1 * dy])
        clipped_p2 = np.array([x0 + u2 * dx, y0 + u2 * dy])
        return clipped_p1, clipped_p2

    @staticmethod
    def compute_circumcenter(A, B, C):
        """
        Computes the circumcenter of a triangle defined by points A, B, and C.
        Returns [np.nan, np.nan] if the points are collinear.
        """
        D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
        if abs(D) < 1e-9:
            return np.array([np.nan, np.nan])

        Ux = (
            (A[0] ** 2 + A[1] ** 2) * (B[1] - C[1])
            + (B[0] ** 2 + B[1] ** 2) * (C[1] - A[1])
            + (C[0] ** 2 + C[1] ** 2) * (A[1] - B[1])
        ) / D
        Uy = (
            (A[0] ** 2 + A[1] ** 2) * (C[0] - B[0])
            + (B[0] ** 2 + B[1] ** 2) * (A[0] - C[0])
            + (C[0] ** 2 + C[1] ** 2) * (B[0] - A[0])
        ) / D
        return np.array([Ux, Uy])

    @staticmethod
    def compute_bezier_control_point(P_A, P_B, M_vor):
        """
        Calculates the control point C for a quadratic Bezier curve starting at P_A,
        ending at P_B, and passing exactly through M_vor at t=0.5.

        Formula: B(0.5) = 0.25*P_A + 0.5*C + 0.25*P_B = M_vor
        """
        return 2 * M_vor - 0.5 * (P_A + P_B)

    @staticmethod
    def get_bezier_trajectory(P_A, P_B, C, num_points=50):
        """
        Generates the points along a quadratic Bezier curve given endpoints and control point.
        """
        t_vals = np.linspace(0, 1, num_points)
        curve_x = (1 - t_vals) ** 2 * P_A[0] + 2 * (1 - t_vals) * t_vals * C[0] + t_vals**2 * P_B[0]
        curve_y = (1 - t_vals) ** 2 * P_A[1] + 2 * (1 - t_vals) * t_vals * C[1] + t_vals**2 * P_B[1]
        return curve_x, curve_y


class RenderQueue:
    """
    Manages structured layout queues and handles strict layered drawing order.
    Supports forwarding custom Matplotlib style kwargs.
    """

    def __init__(self):
        self.faces = []
        self.edges = []
        self.vertices = []

    def clear(self):
        self.faces.clear()
        self.edges.clear()
        self.vertices.clear()

    def queue_face(self, item_type, data, color, alpha, ls="-", label=None, **kwargs):
        self.faces.append(
            {
                "type": item_type,
                "data": data,
                "color": color,
                "alpha": alpha,
                "ls": ls,
                "label": label,
                "style_kwargs": kwargs,
            }
        )

    def queue_edge(self, x, y, ls, color, alpha, label=None, **kwargs):
        self.edges.append(
            {
                "x": x,
                "y": y,
                "ls": ls,
                "color": color,
                "alpha": alpha,
                "label": label,
                "style_kwargs": kwargs,
            }
        )

    def queue_vertex(self, x, y, marker, color, label=None, **kwargs):
        self.vertices.append(
            {
                "x": x,
                "y": y,
                "marker": "marker" if marker == "s" else "o",
                "real_marker": marker,
                "color": color,
                "label": label,
                "style_kwargs": kwargs,
            }
        )

    def render_all(self, ax):
        """
        Executes rendering in a strict layer hierarchy:
        1. Faces (zorder=2)
        2. Edges (zorder=4)
        3. Vertices (zorder=10)
        """
        # 1. Draw Faces
        for item in self.faces:
            alpha = item.get("alpha", 1.0)
            style = item.get("style_kwargs", {})

            if item["type"] == "polygon":
                poly_kwargs = {"facecolor": item["color"], "alpha": alpha, "zorder": 2, "label": item["label"]}
                poly_kwargs.update(style)
                poly = plt.Polygon(item["data"], **poly_kwargs)
                ax.add_patch(poly)

            elif item["type"] == "path":
                patch_kwargs = {
                    "facecolor": item["color"],
                    "alpha": alpha,
                    "edgecolor": item["color"],
                    "lw": 2,
                    "linestyle": item["ls"],
                    "zorder": 2,
                    "label": item["label"],
                }
                if "linewidth" in style:
                    patch_kwargs.pop("lw", None)
                if "lw" in style:
                    patch_kwargs.pop("linewidth", None)
                if "linestyle" in style:
                    patch_kwargs.pop("ls", None)
                if "ls" in style:
                    patch_kwargs.pop("linestyle", None)

                patch_kwargs.update(style)
                patch = patches.PathPatch(item["data"], **patch_kwargs)
                ax.add_patch(patch)

        # 2. Draw Edges
        for item in self.edges:
            alpha = item.get("alpha", 1.0)
            style = item.get("style_kwargs", {})

            edge_kwargs = {
                "linestyle": item["ls"],
                "color": item["color"],
                "linewidth": 4,
                "alpha": alpha,
                "zorder": 4,
                "label": item["label"],
            }
            if "linewidth" in style:
                edge_kwargs.pop("lw", None)
            if "lw" in style:
                edge_kwargs.pop("linewidth", None)
            if "linestyle" in style:
                edge_kwargs.pop("ls", None)
            if "ls" in style:
                edge_kwargs.pop("linestyle", None)

            edge_kwargs.update(style)
            ax.plot(item["x"], item["y"], **edge_kwargs)

        # 3. Draw Vertices (Guaranteed on top)
        for item in self.vertices:
            style = item.get("style_kwargs", {})

            vert_kwargs = {
                "marker": item["real_marker"],
                "color": item["color"],
                "markersize": 10,
                "zorder": 10,
                "label": item["label"],
            }
            if "markersize" in style:
                vert_kwargs.pop("ms", None)
            if "ms" in style:
                vert_kwargs.pop("markersize", None)

            vert_kwargs.update(style)
            ax.plot(item["x"], item["y"], **vert_kwargs)


class DualMesh:
    def __init__(self, points):
        """
        Orchestrates topological relations between Delaunay and Voronoi elements.
        """
        self.points = np.array(points)
        self.tri = Delaunay(self.points)
        self.vor = Voronoi(self.points)

        # Public element properties
        self.tri0, self.tri1, self.tri2 = None, None, None
        self.vor0, self.vor1, self.vor2 = None, None, None

        self.tri1_ctrl = None
        self.edge_ctrl_map = {}

        # Mutable color databases
        self.colors0 = []
        self.colors1 = []
        self.colors2 = []

        # State tracking and delegators
        self.fig = None
        self.ax = None
        self.added_labels = set()
        self.queue = RenderQueue()

        self._build_topology()
        self._init_colors()

    def _build_topology(self):
        # 1. Build Tri1 (Delaunay Edges) & Vor1 (Voronoi Ridges)
        self.tri1_indices = self.vor.ridge_points
        self.tri1 = self.points[self.tri1_indices]
        self.vor1 = []
        self.tri1_ctrl = []

        center = self.points.mean(axis=0)
        far_bound = 10.0

        for (p1_idx, p2_idx), ridge_verts in zip(self.vor.ridge_points, self.vor.ridge_vertices):
            # Resolve Voronoi infinite rays
            if -1 not in ridge_verts:
                coords = [self.vor.vertices[ridge_verts[0]], self.vor.vertices[ridge_verts[1]]]
            else:
                finite_idx = ridge_verts[1] if ridge_verts[0] == -1 else ridge_verts[0]
                finite_pt = self.vor.vertices[finite_idx]

                p1, p2 = self.points[p1_idx], self.points[p2_idx]
                tangent = (p2 - p1) / np.linalg.norm(p2 - p1)
                normal = np.array([-tangent[1], tangent[0]])

                midpoint = (p1 + p2) / 2.0
                if np.dot(midpoint - center, normal) < 0.0:
                    normal = -normal

                far_pt = finite_pt + normal * far_bound
                coords = [finite_pt, far_pt]

            self.vor1.append(np.array(coords))

            # Solve curved Delaunay Bezier alignment
            P_A, P_B = self.points[p1_idx], self.points[p2_idx]
            clipped_vor = GeometryCore.liang_barsky_clip(coords[0], coords[1])

            M_vor = (P_A + P_B) / 2.0 if clipped_vor is None else (clipped_vor[0] + clipped_vor[1]) / 2.0
            C = GeometryCore.compute_bezier_control_point(P_A, P_B, M_vor)

            self.tri1_ctrl.append(C)
            self.edge_ctrl_map[frozenset([p1_idx, p2_idx])] = C

        self.tri1_ctrl = np.array(self.tri1_ctrl)

        # 2. Build Tri0 (Points) & Vor2 (Regions)
        self.tri0 = self.points
        self.vor2 = []

        for i in range(len(self.tri0)):
            region_pts = []
            for (p1_idx, p2_idx), ridge_coords in zip(self.vor.ridge_points, self.vor1):
                if i == p1_idx or i == p2_idx:
                    region_pts.extend(ridge_coords)

            region_pts = np.array(region_pts)

            if len(region_pts) >= 3:
                try:
                    hull = ConvexHull(region_pts)
                    self.vor2.append(region_pts[hull.vertices])
                except Exception:
                    self.vor2.append(region_pts)
            else:
                self.vor2.append(region_pts)

        # 3. Build Tri2 (Triangles) & Vor0 (Vertices)
        self.tri2_indices = self.tri.simplices
        self.tri2 = self.points[self.tri2_indices]
        self.vor0 = []

        for simplex in self.tri2_indices:
            cc = GeometryCore.compute_circumcenter(
                self.points[simplex[0]],
                self.points[simplex[1]],
                self.points[simplex[2]],
            )
            self.vor0.append(cc)

        self.vor0 = np.array(self.vor0)

    def _init_colors(self):
        palette = plt.cm.tab20.colors
        self.colors0 = [palette[i % len(palette)] for i in range(len(self.tri0))]
        self.colors1 = ["k" for _ in range(len(self.tri1))]
        self.colors2 = [palette[i % len(palette)] for i in range(len(self.tri2))]

    def init_plot(self, mode="both", ax=None):
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
        else:
            self.ax = ax
            self.fig = ax.figure

        self.added_labels = set()
        self.queue.clear()

        # Background reference layout (drawn directly on base layer)
        if mode in ["both", "tri"]:
            for i in range(len(self.tri1)):
                curve_x, curve_y = GeometryCore.get_bezier_trajectory(
                    self.tri1[i, 0],
                    self.tri1[i, 1],
                    self.tri1_ctrl[i],
                    num_points=20,
                )
                self.ax.plot(curve_x, curve_y, color="gray", alpha=0.15, linestyle="--", zorder=1)

        if mode in ["both", "vor"]:
            for ridge in self.vor1:
                self.ax.plot(ridge[:, 0], ridge[:, 1], color="gray", alpha=0.3, linestyle="-", zorder=1)

        self.ax.plot(self.points[:, 0], self.points[:, 1], "o", color="gray", markersize=3, alpha=0.2, zorder=1)

    def attach(self, dim=None, index=None, mode="both", color=None, ax=None, alpha=1.0, **kwargs):
        """
        Attaches elements of a specific dimension to the plot.
        Accepts Matplotlib styling arguments via **kwargs and forwards them down to the plotting functions.
        """
        if ax is not None and self.ax is not ax:
            self.init_plot(mode=mode, ax=ax)
        elif self.ax is None:
            self.init_plot(mode=mode)

        if dim is None:
            for d in [0, 1, 2]:
                self.attach(dim=d, index=None, mode=mode, color=color, ax=ax, alpha=alpha, **kwargs)
            return

        indices = (
            [index]
            if index is not None
            else range(len(self.tri0) if dim == 0 else len(self.tri1) if dim == 1 else len(self.tri2))
        )

        for idx in indices:
            item_color = color
            if item_color is None:
                item_color = self.colors0[idx] if dim == 0 else self.colors1[idx] if dim == 1 else self.colors2[idx]

            if dim == 0:
                if mode in ["both", "tri"]:
                    target_pt = self.tri0[idx]
                    label_pt = f"tri0[{idx}] (Point)"
                    is_new = label_pt not in self.added_labels
                    if is_new:
                        self.added_labels.add(label_pt)
                    self.queue.queue_vertex(
                        target_pt[0],
                        target_pt[1],
                        "s",
                        item_color,
                        label_pt if is_new else None,
                        **kwargs,
                    )

                if mode in ["both", "vor"]:
                    target_region = self.vor2[idx]
                    if len(target_region) >= 3:
                        label_face = f"vor2[{idx}] (Face)"
                        is_new = label_face not in self.added_labels
                        if is_new:
                            self.added_labels.add(label_face)

                        self.queue.queue_face(
                            "polygon",
                            target_region,
                            item_color,
                            alpha,
                            label=label_face if is_new else None,
                            **kwargs,
                        )

                        closed_border = np.vstack((target_region, target_region[0]))
                        self.queue.queue_edge(closed_border[:, 0], closed_border[:, 1], "-", item_color, alpha, **kwargs)

            elif dim == 1:
                if mode in ["both", "tri"]:
                    target_tri_edge = self.tri1[idx]
                    C = self.tri1_ctrl[idx]
                    label_tri1 = f"tri1[{idx}] (Curved Edge)"
                    is_new = label_tri1 not in self.added_labels
                    if is_new:
                        self.added_labels.add(label_tri1)

                    curve_x, curve_y = GeometryCore.get_bezier_trajectory(target_tri_edge[0], target_tri_edge[1], C)
                    self.queue.queue_edge(
                        curve_x,
                        curve_y,
                        "--",
                        item_color,
                        alpha,
                        label_tri1 if is_new else None,
                        **kwargs,
                    )

                if mode in ["both", "vor"]:
                    target_vor_edge = self.vor1[idx]
                    label_vor1 = f"vor1[{idx}] (Ridge)"
                    is_new = label_vor1 not in self.added_labels
                    if is_new:
                        self.added_labels.add(label_vor1)

                    self.queue.queue_edge(
                        target_vor_edge[:, 0],
                        target_vor_edge[:, 1],
                        "-",
                        item_color,
                        alpha,
                        label_vor1 if is_new else None,
                        **kwargs,
                    )

            elif dim == 2:
                if mode in ["both", "tri"]:
                    v0, v1, v2 = self.tri2_indices[idx]
                    P0, P1, P2 = self.points[v0], self.points[v1], self.points[v2]

                    C01 = self.edge_ctrl_map[frozenset([v0, v1])]
                    C12 = self.edge_ctrl_map[frozenset([v1, v2])]
                    C20 = self.edge_ctrl_map[frozenset([v2, v0])]

                    verts = [P0, C01, P1, C12, P2, C20, P0]
                    codes = [
                        Path.MOVETO,
                        Path.CURVE3,
                        Path.CURVE3,
                        Path.CURVE3,
                        Path.CURVE3,
                        Path.CURVE3,
                        Path.CURVE3,
                    ]
                    path = Path(verts, codes)

                    label_tri2 = f"tri2[{idx}] (Curved Triangle)"
                    is_new = label_tri2 not in self.added_labels
                    if is_new:
                        self.added_labels.add(label_tri2)

                    self.queue.queue_face(
                        "path",
                        path,
                        item_color,
                        alpha,
                        ls="--",
                        label=label_tri2 if is_new else None,
                        **kwargs,
                    )

                if mode in ["both", "vor"]:
                    target_vor_pt = self.vor0[idx]
                    label_vor0 = f"vor0[{idx}] (Vertex)"
                    is_new = label_vor0 not in self.added_labels
                    if is_new:
                        self.added_labels.add(label_vor0)

                    self.queue.queue_vertex(
                        target_vor_pt[0],
                        target_vor_pt[1],
                        "o",
                        item_color,
                        label_vor0 if is_new else None,
                        **kwargs,
                    )

    def show(self, legend=True, plt_show=True):
        if self.ax is None:
            return

        # Render layered drawings through the RenderQueue
        self.queue.render_all(self.ax)

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect("equal")

        if legend and self.added_labels:
            ncols = 3 if len(self.added_labels) > 30 else 2 if len(self.added_labels) > 15 else 1
            font_sz = (
                "xx-small"
                if len(self.added_labels) > 30
                else "x-small"
                if len(self.added_labels) > 15
                else "small"
            )
            self.ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=font_sz, ncol=ncols)

        self.ax.set_title("Dual Mesh Visualization", fontsize=14)

        if plt_show:
            plt.tight_layout()
            plt.show()

        # Clean state
        self.fig = None
        self.ax = None
        self.added_labels.clear()
        self.queue.clear()


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    random_points = np.random.rand(8, 2)
    mesh = DualMesh(random_points)

    # Subplot execution test (Multi-ax layout with custom styles)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Left subplot: Voronoi with thin lines
    mesh.attach(ax=axs[0], mode="vor", lw=1.5)
    mesh.show(legend=False, plt_show=False)

    # Right subplot: Delaunay with dashed curved edges and custom thickness
    mesh.attach(ax=axs[1], mode="tri", linewidth=1.5)
    mesh.show(legend=False, plt_show=False)

    plt.tight_layout()
    plt.show()

