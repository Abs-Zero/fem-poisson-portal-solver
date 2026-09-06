import gmsh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import cg


class PortalPoissonSolver:
    def __init__(
        self,
        portals,
        domain_side_length=2.0,
        mesh_size_regular=0.05,
        mesh_size_min=0.005,
        switch_portal=True,
    ):
        # portals: 所有传送门及其位置
        # [[(1x坐标, 1y坐标, 1长度, 1正方向), (2x坐标, 2y坐标, 2长度, 2正方向), 是否翻转], ...]
        self.portals = portals
        self.domain_side_length = domain_side_length
        self.mesh_size_regular = mesh_size_regular
        self.mesh_size_min = mesh_size_min
        self.switch_portal = switch_portal  # 是否使用传送门
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

    def _distribution_func(self, size_min, size_max, portal):  # 传送门上节点分布
        line1, line2, _ = portal
        _, _, l1, _ = line1
        _, _, l2, _ = line2
        x11, y11, x12, y12, x21, y21, x22, y22 = self._calculate_portal_local(portal)
        dx1, dy1 = [(x12 - x11) / l1, (y12 - y11) / l1]
        dx2, dy2 = [(x22 - x21) / l2, (y22 - y21) / l2]

        half_distance = (l1 + l2) / 4

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

    def generate_domain_mesh(self, domain_shape="rectangle"):  # 生成网格
        if self.mesh_size_regular < self.mesh_size_min:
            raise ValueError("网格尺寸不合理")
        print("生成网格...")

        def _add_points_to_mesh(points):  #  将点列添加到网格
            n = len(points)
            point_tags = [0] * n
            line_tags = [0] * (n - 1)
            for i, (x, y) in enumerate(points):
                point_tags[i] = gmsh.model.occ.addPoint(x, y, 0, self.mesh_size_regular)
            for i in range(n - 1):
                line_tags[i] = gmsh.model.occ.addLine(point_tags[i], point_tags[i + 1])
            return point_tags, line_tags

        def _add_portal_to_mesh(portal):  #  将一对传送门添加到网格
            p1, p2 = self._distribution_func(
                self.mesh_size_min, self.mesh_size_regular, portal
            )
            p1_tags, l1_tags = _add_points_to_mesh(p1)
            p2_tags, l2_tags = _add_points_to_mesh(p2)
            return p1_tags, p2_tags, l1_tags + l2_tags

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add("main_mesh")
        D = self.domain_side_length

        if domain_shape == "rectangle":
            domain = gmsh.model.occ.addRectangle(-D, -D, 0, 2 * D, 2 * D)
        elif domain_shape == "circle":
            circle = gmsh.model.occ.addCircle(0, 0, 0, D)
            loop = gmsh.model.occ.addCurveLoop([circle])
            domain = gmsh.model.occ.addPlaneSurface([loop])
        else:
            raise ValueError("定义域形状错误")

        portal_points_tags = []
        line_tags = []
        for portal in self.portals:
            p1_tags, p2_tags, l_tags = _add_portal_to_mesh(portal)
            portal_points_tags.append((p1_tags, p2_tags))
            line_tags.extend((1, lt) for lt in l_tags)

        try:
            if line_tags:
                outDimTags, outMap = gmsh.model.occ.fragment([(2, domain)], line_tags)
            gmsh.model.occ.synchronize()

            if line_tags:
                for lt in line_tags:
                    gmsh.model.mesh.setTransfiniteCurve(lt[1], 1)
        except Exception as e:
            print(e)
            print("网格生成错误，请检查传送门是否重叠")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.mesh_size_regular)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.mesh_size_min)
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        self.nodes = np.array(node_coords).reshape(-1, 3)[:, :2]

        tag2idx = {int(t): i for i, t in enumerate(node_tags)}

        _, _, node_tags = gmsh.model.mesh.getElements(2, domain)
        self.elements = np.array(node_tags[0], dtype=int).reshape(-1, 3) - 1
        self.elements_unswitched = self.elements.copy()

        for i, portal in enumerate(self.portals):
            # 传送门上节点标签 -> 数组索引
            line1, line2, _ = portal
            x, y, _, theta = zip(line1, line2)
            p1, p2 = portal_points_tags[i]
            p1, p2 = p1[1:-1], p2[1:-1]

            for j in range(len(p1)):
                n1, _, _ = gmsh.model.mesh.getNodes(0, p1[j])
                p1[j] = n1[0] - 1
                n2, _, _ = gmsh.model.mesh.getNodes(0, p2[j])
                p2[j] = n2[0] - 1
            self.portal_points[i] = (p1, p2)

        def _tri_centroid_xy(a, b, c):
            x1, y1 = self.nodes[a]
            x2, y2 = self.nodes[b]
            x3, y3 = self.nodes[c]
            return (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3

        if self.switch_portal == True:
            for i, (p1, p2) in enumerate(self.portal_points):
                line1, line2, _ = self.portals[i]
                x, y, _, theta = zip(line1, line2)
                p1, p2 = p1.copy(), p2.copy()

                inner_nodes = {n: 0 for n in p1} | {n: 1 for n in p2}
                inner_nodes_map = {a: b for a, b in zip(p1, p2)} | {
                    b: a for a, b in zip(p1, p2)
                }
                switch_rule = set()
                for j, Ele in enumerate(self.elements):
                    for k, node_tag in enumerate(Ele):
                        if node_tag in inner_nodes:
                            p_id = inner_nodes[node_tag]
                            a, b, c = Ele

                            tri_x, tri_y = _tri_centroid_xy(a, b, c)

                            co, si = np.cos(theta[p_id]), np.sin(theta[p_id])
                            dx, dy = tri_x - x[p_id], tri_y - y[p_id]
                            if co * dx + si * dy < 0:
                                st, ed = node_tag, inner_nodes_map[node_tag]
                                switch_rule.add((j, k, st, ed))

                for j, k, st, ed in switch_rule:
                    self.elements[j][k] = ed
                    self.switched_portal_element_points[i].setdefault(j, set()).add(
                        (k, st, ed)
                    )

        boundary_curves = gmsh.model.getBoundary(
            [(2, domain)], oriented=False, recursive=False
        )
        boundary_entities = gmsh.model.getBoundary(
            [(2, domain)], oriented=False, recursive=True
        )
        bnd_node_tags = set()
        for dim, ctag in boundary_curves + boundary_entities:
            ntags, _, _ = gmsh.model.mesh.getNodes(dim, ctag)
            for t in ntags:
                bnd_node_tags.add(int(t))
        self.boundary_nodes = np.array([tag2idx[t] for t in bnd_node_tags], dtype=int)

        print(f"节点数量: {len(self.nodes)}, 面元数量: {len(self.elements)}")
        gmsh.finalize()

    def plot_mesh(
        self,
        show_ele_tags=False,
        plot_boundary_nodes=True,
        plot_portal_points=True,
        fig_name="meshfig.svg",
    ):  # 绘制网格
        if not hasattr(self, "elements"):
            self.generate_domain_mesh()
        print("绘制网格图像...")
        D = self.domain_side_length
        for i, (a, b, c) in enumerate(self.elements):
            x1, y1 = self.nodes[a]
            x2, y2 = self.nodes[b]
            x3, y3 = self.nodes[c]
            x = (x1 + x2 + x3) / 3
            y = (y1 + y2 + y3) / 3
            plt.plot([x1, x2, x3, x1], [y1, y2, y3, y1], color="r", linewidth=0.3)

            area = 0.5 * abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            font = np.sqrt(area) / D * 40
            if show_ele_tags:
                plt.text(x, y, f"{i}", fontsize=font, ha="center", va="center")

        if plot_boundary_nodes:
            plt.scatter(
                self.nodes[self.boundary_nodes, 0],
                self.nodes[self.boundary_nodes, 1],
                color="b",
                s=20,
            )

        if plot_portal_points:
            xs, ys = [], []
            for p1s, p2s in self.portal_points:
                for t1, t2 in zip(p1s, p2s):
                    for t in (t1, t2):
                        xs.append(self.nodes[t][0])
                        ys.append(self.nodes[t][1])
            plt.plot(xs, ys, "ko", markersize=1)

        plt.axis("equal")
        plt.savefig(fig_name)
        plt.show()

    def solve_poisson(self, func_domain, solve_method="cg"):  # 求解方程
        # Galerkin 法
        if not hasattr(self, "elements"):
            self.generate_domain_mesh()
        print("求解...")

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
        draw_contours=True,
        plot_portal=True,
        fig_name="meshfig.svg",
    ):  # 绘制解图像
        if not hasattr(self, "solution"):
            print("尚未求解，请先调用 solve_poisson")
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
        if draw_contours:
            plt.tricontourf(triang, solution, levels=50)
        else:
            plt.tripcolor(triang, solution)
        plt.colorbar()

        if plot_portal:
            for portal in self.portals:
                x11, y11, x12, y12, x21, y21, x22, y22 = self._calculate_portal_local(
                    portal
                )

                plt.plot([x11, x12], [y11, y12], color="black", linewidth=0.2)
                plt.plot([x21, x22], [y21, y22], color="black", linewidth=0.2)

        plt.axis("equal")
        plt.savefig(fig_name)
        plt.show()


if __name__ == "__main__":
    D = 5
    MS_REGULAR = 0.5
    MS_MIN = 0.2

    portals = [
        [(1, 0, 2, np.pi), (-1, 0, 2, np.pi), True],
        [(0, 1, 2, np.pi / 2), (0, -1, 2, np.pi / 2), True],
    ]

    solver = PortalPoissonSolver(portals, D, MS_REGULAR, MS_MIN)
    solver.generate_domain_mesh()
    solver.plot_mesh(
        show_ele_tags=False,
        plot_boundary_nodes=True,
        plot_portal_points=True,
        fig_name="meshfig.svg",
    )

    def func_domain_source(x, y):
        xc, yc = 0, 0
        dx, dy = x - xc, y - yc
        r = np.sqrt((dx) ** 2 + (dy) ** 2)
        # r = abs(dx)+abs(dy)
        # r = max(abs(dx), abs(dy))
        if r < 0.5:
            return 100
        else:
            return 0

    solver.solve_poisson(func_domain_source)
    solver.plot_solution(draw_contours=False, plot_portal=True)
    # print(solver.evaluate(1, 1))
