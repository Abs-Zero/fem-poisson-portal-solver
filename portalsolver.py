import gmsh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg
import os

class PortalPoissonSolver:
    def __init__(
        self,
        portals,
        domain_side_length: float | tuple = 2.0,
        mesh_size_regular=0.05,
        mesh_size_min=0.005,
        switch_portal=True,
        show_progress=True,
    ):
        # portals: 所有传送门及其位置
        # [[(1x坐标, 1y坐标, 1长度, 1正方向), (2x坐标, 2y坐标, 2长度, 2正方向), 是否翻转], ...]
        self.portals = portals
        self.domain_side_length = domain_side_length
        self.mesh_size_regular = mesh_size_regular
        self.mesh_size_min = mesh_size_min
        self.switch_portal = switch_portal  # 是否使用传送门
        self.show_progress = show_progress  # 是否输出求解进度
        self.portal_points = [(np.array([]), np.array([])) for _ in portals]
        # switched_portal_element_points: 传送门节点变换记录
        # [{(面元 -> (顶点, 初始节点, 目标节点), ...}, ...]
        self.switched_portal_element_points = [{} for _ in portals]
        self.nodes: np.ndarray
        self.boundary_nodes: np.ndarray
        self.elements: np.ndarray
        self.elements_unswitched: np.ndarray
        self.solution: np.ndarray

    def _calculate_portal_local(self, portal):  # 传送门端点位置
        line1, line2, portal_reverse = portal
        xm1, ym1, l1, theta1 = line1
        xm2, ym2, l2, theta2 = line2
        alpha1 = theta1 + np.pi / 2
        alpha2 = theta2 + np.pi / 2
        x11, y11, x12, y12 = (
            xm1 - l1 / 2 * np.cos(alpha1),
            ym1 - l1 / 2 * np.sin(alpha1),
            xm1 + l1 / 2 * np.cos(alpha1),
            ym1 + l1 / 2 * np.sin(alpha1),
        )
        sign = -1 if portal_reverse else 1
        x21, y21, x22, y22 = (
            xm2 - sign * l2 / 2 * np.cos(alpha2),
            ym2 - sign * l2 / 2 * np.sin(alpha2),
            xm2 + sign * l2 / 2 * np.cos(alpha2),
            ym2 + sign * l2 / 2 * np.sin(alpha2),
        )

        return x11, y11, x12, y12, x21, y21, x22, y22

    def _distribution_func(self, portal):  # 传送门上节点分布
        line1, line2, _ = portal
        _, _, l1, _ = line1
        _, _, l2, _ = line2
        x11, y11, x12, y12, x21, y21, x22, y22 = self._calculate_portal_local(portal)
        dx1, dy1 = [(x12 - x11) / l1, (y12 - y11) / l1]
        dx2, dy2 = [(x22 - x21) / l2, (y22 - y21) / l2]

        l_argv = (l1 + l2) / 2

        half_distance = l_argv / 2

        size_min = min(l_argv / 50, self.mesh_size_min)
        size_max = min(l_argv / 5, self.mesh_size_regular)

        def _base_distribution_func(x):
            return np.sin(np.pi / 2 * x) ** 2

        def _total_segment_length(size_min, size_max, num_segs):
            d = 0
            for i in range(num_segs):
                d += (
                    _base_distribution_func(i / num_segs) * (size_max - size_min)
                    + size_min
                )
            return d

        num_segs_min = int(half_distance / size_max) - 1
        num_segs_max = int(half_distance / size_min) + 1
        while True:
            num_segs = (num_segs_min + num_segs_max) // 2
            d = _total_segment_length(size_min, size_max, num_segs)
            if (
                _total_segment_length(size_min, size_max, num_segs - 1)
                <= half_distance
                <= d
            ):
                break
            elif (
                _total_segment_length(size_min, size_max, num_segs + 1)
                >= half_distance
                >= d
            ):
                num_segs += 1
                break
            elif d > half_distance:
                num_segs_max = num_segs
            else:
                num_segs_min = num_segs
        interp_nodes = [0.0] * (2 * num_segs + 1)
        for i in range(num_segs - 1):
            interp_nodes[i + 1] = (
                interp_nodes[i]
                + _base_distribution_func(i / num_segs) * (size_max - size_min)
                + size_min
            )
        interp_nodes[num_segs] = half_distance
        for i in range(num_segs + 1, 2 * num_segs + 1):
            interp_nodes[i] = 2 * half_distance - interp_nodes[2 * num_segs - i]

        p1 = [(0, 0)] * len(interp_nodes)
        p2 = [(0, 0)] * len(interp_nodes)
        for i, d in enumerate(interp_nodes):
            d /= half_distance * 2
            d1 = d * l1
            d2 = d * l2
            p1[i] = (x11 + dx1 * d1, y11 + dy1 * d1)
            p2[i] = (x21 + dx2 * d2, y21 + dy2 * d2)

        return p1, p2

    def generate_domain_mesh(self, domain_shape="rectangle"):
        if self.mesh_size_regular < self.mesh_size_min:
            raise ValueError("网格尺寸不合理")
        if self.show_progress:
            print("生成网格...")

        def _add_points_to_mesh(points):
            n = len(points)
            point_tags = [0] * n
            line_tags = [0] * (n - 1)
            for i, (x, y) in enumerate(points):
                point_tags[i] = gmsh.model.occ.addPoint(x, y, 0, self.mesh_size_regular)
            for i in range(n - 1):
                line_tags[i] = gmsh.model.occ.addLine(point_tags[i], point_tags[i + 1])
            return point_tags, line_tags

        def _add_portal_to_mesh(portal):
            p1, p2 = self._distribution_func(portal)
            p1_tags, l1_tags = _add_points_to_mesh(p1)
            p2_tags, l2_tags = _add_points_to_mesh(p2)
            return p1_tags, p2_tags, l1_tags + l2_tags

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add("main_mesh")
        D = self.domain_side_length

        if domain_shape == "rectangle":
            if isinstance(self.domain_side_length, (int, float)):
                A = B = self.domain_side_length
            else:
                A, B = self.domain_side_length
            domain = gmsh.model.occ.addRectangle(-A, -B, 0, 2 * A, 2 * B)
            domain_dimtag = (2, domain)
        elif domain_shape == "circle":
            D = self.domain_side_length
            circle = gmsh.model.occ.addCircle(0, 0, 0, D)
            loop = gmsh.model.occ.addCurveLoop([circle])
            domain = gmsh.model.occ.addPlaneSurface([loop])
            domain_dimtag = (2, domain)
        else:
            raise ValueError("定义域形状错误")

        portal_points_tags = []  # [(p1_pointTags, p2_pointTags), ...]
        line_dimtags = []  # [(1, lineTag), ...]

        for portal in self.portals:
            p1_tags, p2_tags, l_tags = _add_portal_to_mesh(portal)
            portal_points_tags.append((p1_tags, p2_tags))
            line_dimtags.extend((1, lt) for lt in l_tags)

        outDimTags = [domain_dimtag]
        outMap = None
        try:
            if line_dimtags:
                outDimTags, outMap = gmsh.model.occ.fragment(
                    [domain_dimtag], line_dimtags
                )
            gmsh.model.occ.synchronize()

            if line_dimtags:
                for dim, lt in line_dimtags:
                    gmsh.model.mesh.setTransfiniteCurve(lt, 1)
        except Exception as e:
            print(e)
            raise ValueError("网格生成错误，请检查传送门是否交叉")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.mesh_size_regular)
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        self.nodes = np.array(node_coords).reshape(-1, 3)[:, :2]
        tag2idx = {int(t): i for i, t in enumerate(node_tags)}

        solve_surfs = [dt for dt in outDimTags if dt[0] == 2]

        all_tris = []
        for dim, stag in solve_surfs:
            etypes, etags, enodes = gmsh.model.mesh.getElements(dim, stag)
            if not enodes:
                continue
            tris = np.array(enodes[0]).reshape(-1, 3)
            all_tris.append(tris)

        self.elements_unswitched = np.vectorize(tag2idx.get)(np.vstack(all_tris))
        self.elements = self.elements_unswitched.copy()

        self.portal_points = [(np.array([]), np.array([])) for _ in self.portals]

        for i, portal in enumerate(self.portals):
            p1_occ_pts, p2_occ_pts = portal_points_tags[i]
            # p1_occ_pts = p1_occ_pts[1:-1]
            # p2_occ_pts = p2_occ_pts[1:-1]
            try:
                p1_idx, p2_idx = [], []
                for pt in p1_occ_pts:
                    ntags, _, _ = gmsh.model.mesh.getNodes(0, pt)
                    if len(ntags) > 0:
                        p1_idx.append(tag2idx[int(ntags[0])])
                for pt in p2_occ_pts:
                    ntags, _, _ = gmsh.model.mesh.getNodes(0, pt)
                    if len(ntags) > 0:
                        p2_idx.append(tag2idx[int(ntags[0])])
            except Exception as e:
                print(e)
                raise ValueError("网格生成错误，请检查传送门是否重叠")

            self.portal_points[i] = (np.array(p1_idx), np.array(p2_idx))

        def _tri_centroid_xy(a, b, c):
            x1, y1 = self.nodes[a]
            x2, y2 = self.nodes[b]
            x3, y3 = self.nodes[c]
            return (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3

        self.switched_portal_element_points = [{} for _ in self.portals]
        if self.switch_portal:
            for i, (p1, p2) in enumerate(self.portal_points):
                line1, line2, _ = self.portals[i]
                x, y, _, theta = zip(line1, line2)

                inner_nodes = {int(n): 0 for n in p1} | {int(n): 1 for n in p2}
                inner_nodes_map = {int(a): int(b) for a, b in zip(p1, p2)} | {
                    int(b): int(a) for a, b in zip(p1, p2)
                }

                switch_rule = set()
                for j, Ele in enumerate(self.elements):
                    for k, node_idx in enumerate(Ele):
                        node_idx = int(node_idx)
                        if node_idx in inner_nodes:
                            p_id = inner_nodes[node_idx]
                            a, b, c = map(int, Ele)

                            tri_x, tri_y = _tri_centroid_xy(a, b, c)
                            co, si = np.cos(theta[p_id]), np.sin(theta[p_id])
                            dx, dy = tri_x - x[p_id], tri_y - y[p_id]
                            if co * dx + si * dy < 0:
                                st, ed = node_idx, inner_nodes_map[node_idx]
                                switch_rule.add((j, k, st, ed))

                for j, k, st, ed in switch_rule:
                    self.elements[j][k] = ed
                    self.switched_portal_element_points[i].setdefault(j, set()).add(
                        (k, st, ed)
                    )

        boundary_curves = gmsh.model.getBoundary(
            solve_surfs, oriented=False, recursive=False
        )
        boundary_points = gmsh.model.getBoundary(
            solve_surfs, oriented=False, recursive=True
        )

        bnd_node_tags = set()
        for dim, tag in boundary_curves + boundary_points:
            ntags, _, _ = gmsh.model.mesh.getNodes(dim, tag)
            bnd_node_tags.update(map(int, ntags))

        self.boundary_nodes = np.array([tag2idx[t] for t in bnd_node_tags], dtype=int)

        if self.show_progress:
            print(f"节点数量: {len(self.nodes)}, 面元数量: {len(self.elements)}")
        gmsh.finalize()

    def plot_portal(self):
        # fmt: off
        for portal_idx, portal in enumerate(self.portals):
            line1, line2, reverse = portal
            xm1, ym1, l1, theta1 = line1
            xm2, ym2, l2, theta2 = line2
            x11, y11, x12, y12, x21, y21, x22, y22 = self._calculate_portal_local(portal)

            plt.plot([x11, x12], [y11, y12], color="black", linewidth=0.2)
            plt.plot([x21, x22], [y21, y22], color="black", linewidth=0.2)
            
            l_argv = (l1 + l2) / 2
            
            arrow_len = l_argv / 5
            linewidth = arrow_len
            head_width = arrow_len / 6
            head_length = arrow_len / 4
            
            fontsize = l_argv * 2
            dx1, dy1 = arrow_len * np.cos(theta1), arrow_len * np.sin(theta1)
            dx2, dy2 = arrow_len * np.cos(theta2), arrow_len * np.sin(theta2)
            plt.arrow(xm1, ym1, dx1, dy1, head_width=head_width, head_length=head_length, 
                    fc='black', ec='black', linewidth=linewidth)
            plt.arrow(xm2, ym2, dx2, dy2, head_width=head_width, head_length=head_length, 
                    fc='black', ec='black', linewidth=linewidth)
            
            offset = 0.1
            plt.text(xm1+offset, ym1+offset, f'{portal_idx}', 
                    fontsize=fontsize, color='black', ha='center')
            plt.text(xm2+offset, ym2+offset, f'{portal_idx}', 
                    fontsize=fontsize, color='black', ha='center')

            plt.text(x11+offset, y11+offset, '0', fontsize=fontsize, color='black', ha='center')
            plt.text(x12+offset, y12+offset, '1', fontsize=fontsize, color='black', ha='center')
            plt.text(x21+offset, y21+offset, '0', fontsize=fontsize, color='black', ha='center')
            plt.text(x22+offset, y22+offset, '1', fontsize=fontsize, color='black', ha='center')
        # fmt: on

    def plot_mesh(
        self,
        show_ele_tags=False,
        plot_boundary_nodes=True,
        plot_portal_points=True,
        plot_portal=True,
        fig_name="meshfig.svg",
    ):  # 绘制网格
        if not hasattr(self, "elements"):
            self.generate_domain_mesh()
        if self.show_progress:
            print("绘制网格图像...")
        D = self.domain_side_length

        for i, (a, b, c) in enumerate(self.elements):
            x1, y1 = self.nodes[a]
            x2, y2 = self.nodes[b]
            x3, y3 = self.nodes[c]
            x = (x1 + x2 + x3) / 3
            y = (y1 + y2 + y3) / 3

            d1 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            d2 = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
            d3 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            min_d = min(d1, d2, d3)

            linewidth = min_d
            plt.plot(
                [x1, x2, x3, x1],
                [y1, y2, y3, y1],
                color="r",
                linewidth=linewidth,
                zorder=0,
            )
            fontsize = min_d**2 * 5
            if show_ele_tags:
                plt.text(x, y, f"{i}", fontsize=fontsize, ha="center", va="center")

        if plot_boundary_nodes:
            point_size = self.mesh_size_regular * 4
            plt.scatter(
                self.nodes[self.boundary_nodes, 0],
                self.nodes[self.boundary_nodes, 1],
                color="b",
                s=point_size,
            )

        if plot_portal_points:
            xs, ys = [], []
            for p1s, p2s in self.portal_points:
                for t1, t2 in zip(p1s, p2s):
                    for t in (t1, t2):
                        xs.append(self.nodes[t][0])
                        ys.append(self.nodes[t][1])
            plt.plot(xs, ys, "ko", markersize=0.3)

        if plot_portal:
            self.plot_portal()

        plt.axis("equal")
        os.makedirs("./output", exist_ok=True)
        plt.savefig(os.path.join("./output", fig_name))
        plt.show()

    def solve_poisson(self, func_domain, solve_method="cg"):  # 求解方程
        # Galerkin 法
        if not hasattr(self, "elements"):
            self.generate_domain_mesh()
        if self.show_progress:
            print("求解方程...")

        def _fill_KF(self, K, F):
            for idx_ele in range(len(self.elements)):
                id1, id2, id3 = self.elements_unswitched[idx_ele]
                x1, y1 = self.nodes[id1]
                x2, y2 = self.nodes[id2]
                x3, y3 = self.nodes[id3]
                xc = (x1 + x2 + x3) / 3
                yc = (y1 + y2 + y3) / 3
                b = np.array([y2 - y3, y3 - y1, y1 - y2])
                c = np.array([x3 - x2, x1 - x3, x2 - x1])
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))

                A_loc = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        A_loc[i, j] = (b[i] * b[j] + c[i] * c[j]) / (4 * area)

                f_val = func_domain(xc, yc)
                B_loc = area * f_val / 3 * np.ones(3)
                ids = self.elements[idx_ele]

                for i in range(3):
                    F[ids[i]] += B_loc[i]
                    for j in range(3):
                        K[ids[i], ids[j]] += A_loc[i, j]

            for n in self.boundary_nodes:
                K[n, :] = 0
                K[:, n] = 0
                K[n, n] = 1
                F[n] = 0
            return

        num_N = len(self.nodes)
        F = np.zeros(num_N)
        if solve_method == "direct":
            K = np.zeros((num_N, num_N))
            _fill_KF(self, K, F)
            U = np.linalg.solve(K, F)
        elif solve_method == "cg":
            K = lil_matrix((num_N, num_N))
            _fill_KF(self, K, F)
            K_csr = K.tocsr()
            U, _ = cg(K_csr, F, rtol=1e-6, maxiter=5000)
        else:
            raise ValueError("线性求解器名称错误")

        self.solution = U
        return U

    def evaluate(self, x, y):  # 计算 x, y 处解的函数值
        if not hasattr(self, "solution"):
            print("尚未求解，请先调用 solve_poisson")
        if not hasattr(self, "_triang_unswitched"):
            xs, ys = self.nodes[:, 0], self.nodes[:, 1]
            elements_unswitched = self.elements_unswitched
            self._triang_unswitched = mtri.Triangulation(xs, ys, elements_unswitched)
            self._trifinder_unswitched = self._triang_unswitched.get_trifinder()

        tri_id: int = self._trifinder_unswitched(x, y)  # type: ignore
        if tri_id == -1:
            return np.nan
        a, b, c = self.elements_unswitched[tri_id]
        x1, y1 = self.nodes[a]
        x2, y2 = self.nodes[b]
        x3, y3 = self.nodes[c]

        va, vb, vc = self.elements[tri_id]
        u1 = self.solution[va]
        u2 = self.solution[vb]
        u3 = self.solution[vc]

        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
        w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
        w3 = 1.0 - w1 - w2

        return w1 * u1 + w2 * u2 + w3 * u3

    def plot_solution(
        self,
        source_boundary=[],
        draw_contours=True,
        plot_portal=True,
        fig_name="solutionfig.svg",
    ):  # 绘制解图像
        if not hasattr(self, "solution"):
            print("尚未求解，请先调用 solve_poisson")
        if self.show_progress:
            print("绘制解图像...")
        x = list(self.nodes[:, 0].copy())
        y = list(self.nodes[:, 1].copy())
        elements = list(self.elements_unswitched.copy())
        solution = list(self.solution.copy())
        switched_all_portals = self.switched_portal_element_points
        for switched_element in switched_all_portals:
            for j in switched_element:
                for k, st, ed in switched_element[j]:
                    elements[j][k] = len(x)
                    x.append(x[st])
                    y.append(y[st])
                    solution.append(solution[ed])

        triang = mtri.Triangulation(x, y, elements)
        CMAP = "turbo"
        if draw_contours:
            plt.tricontourf(triang, solution, levels=50, cmap=CMAP)
        else:
            plt.tripcolor(triang, solution, cmap=CMAP)
        plt.colorbar()

        if source_boundary:
            for boundary in source_boundary:
                plt.plot(
                    boundary[0],
                    boundary[1],
                    color="black",
                    linewidth=self.mesh_size_min,
                )

        if plot_portal:
            self.plot_portal()

        plt.axis("equal")
        os.makedirs("./output", exist_ok=True)
        plt.savefig(os.path.join("./output", fig_name))
        plt.show()


if __name__ == "__main__":
    portals = [
        [(1, 0, 2, 0), (-1, 0, 2, np.pi / 3), False],
        [(0, 2, 2, np.pi / 2), (0, -2, 2, np.pi / 3), False],
    ]
    solver = PortalPoissonSolver(
        portals,
        domain_side_length=5,
        mesh_size_regular=0.6,
        mesh_size_min=0.3,
        switch_portal=True,
        show_progress=True,
    )
    solver.generate_domain_mesh(domain_shape="circle")  # "rectangle", "circle"

    # 绘制网格图像，使用时网格尺寸不宜过小，否则绘图耗时将会极长
    solver.plot_mesh(
        show_ele_tags=True,
        plot_boundary_nodes=True,
        plot_portal_points=True,
        plot_portal=True,
        fig_name="meshfig.svg",
    )

    xc, yc = 0, 0
    R = 0.3
    G = 100

    def source_func(x, y):
        dx, dy = x - xc, y - yc
        r = np.sqrt((dx) ** 2 + (dy) ** 2)
        # r = abs(dx)+abs(dy)
        # r = max(abs(dx), abs(dy))
        if r < R:
            return G
        else:
            return 0

    source_boundary = []  # 仅用于绘图，非必须参数
    t = np.linspace(0, 2 * np.pi, 100)
    xs, ys = xc + R * np.cos(t), yc + R * np.sin(t)
    source_boundary.append((xs, ys))

    solver.solve_poisson(source_func, solve_method="cg")
    solver.plot_solution(
        source_boundary=source_boundary, draw_contours=True, plot_portal=True
    )

    print(solver.evaluate(1, 1))  # 计算特定点函数值
