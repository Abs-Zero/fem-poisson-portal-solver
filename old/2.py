import gmsh
import math
import csv
from typing import List, Tuple, Dict, Optional

Point3 = Tuple[float, float, float]
PortalPair = Tuple[Point3, Point3, Point3, Point3]  # A0,A1,B0,B1


def seg_len(a: Point3, b: Point3) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def lerp(a: Point3, b: Point3, t: float) -> Point3:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def add_portal_as_polyline(
    P0: Point3,
    P1: Point3,
    t_breaks: List[float],
    point_tag_base: int,
    line_tag_base: int
) -> Tuple[List[int], List[int]]:
    """
    将单条传送门线段 P0->P1 按 t_breaks 切分成多段线（折线仍共线）。
    返回：
      curve_tags: 每段 line 的 tag（按从 P0 到 P1 顺序）
      point_tags: 端点+分割点的 tag（同顺序）
    """
    ts = sorted([t for t in t_breaks if 0.0 < t < 1.0])
    pts: List[int] = []
    # 端点
    p0_tag = gmsh.model.occ.addPoint(*P0, tag=point_tag_base + 0)
    pts.append(p0_tag)
    # 中间点
    for i, t in enumerate(ts):
        p = lerp(P0, P1, t)
        pts.append(gmsh.model.occ.addPoint(*p, tag=point_tag_base + 1 + i))
    # 末端点
    p1_tag = gmsh.model.occ.addPoint(*P1, tag=point_tag_base + 1 + len(ts))
    pts.append(p1_tag)

    # 连成分段线（每段可指定 line tag，便于后续固定识别）
    curves: List[int] = []
    for i in range(len(pts) - 1):
        curves.append(gmsh.model.occ.addLine(pts[i], pts[i + 1], tag=line_tag_base + i))

    return curves, pts


def set_identical_transfinite_for_pair(
    curvesA: List[int],
    curvesB: List[int],
    points_per_segment: int,
    mode: str,
    coef: float
) -> None:
    """
    对一对传送门（已切分后的多段曲线）逐段设置完全相同的 Transfinite。
    points_per_segment = m+1（m 段，m+1 点）
      - 若你想“门上绝不出现额外点”，就设 points_per_segment=2（不细分）
    mode:
      - "Progression", coef=1.0 -> 等间距
      - "Bump", coef>1.0        -> 两端更细（对称），仍严格对应
    """
    if len(curvesA) != len(curvesB):
        raise ValueError("A/B 曲线切分段数不一致，无法逐段严格对应。")

    for la, lb in zip(curvesA, curvesB):
        gmsh.model.mesh.setTransfiniteCurve(la, points_per_segment, mode, coef)
        gmsh.model.mesh.setTransfiniteCurve(lb, points_per_segment, mode, coef)


def curve_nodes_sorted(curve_tag: int) -> Tuple[List[int], List[float]]:
    """
    返回曲线上的节点 tag（按曲线参数排序）与对应参数值。
    """
    node_tags, _, params = gmsh.model.mesh.getNodes(1, curve_tag, includeBoundary=True)
    node_tags = [int(t) for t in node_tags]
    params = [float(p) for p in params]
    if len(node_tags) != len(params):
        raise RuntimeError("gmsh.getNodes 返回的 params 长度异常，无法稳定排序。")
    order = sorted(range(len(node_tags)), key=lambda i: params[i])
    return [node_tags[i] for i in order], [params[i] for i in order]


def build_mapping_and_validate(
    curvesA: List[int],
    curvesB: List[int],
    reverse_B: bool,
    pair_index: int,
    tol_param: float = 0.0
) -> List[Dict[str, int]]:
    """
    逐段提取 A/B 的节点序列并校验严格一致（长度相同、节点数量相同、参数序列一致到 tol_param）。
    返回 mapping 行列表：pair, seg, i, nodeA, nodeB
    """
    if len(curvesA) != len(curvesB):
        raise RuntimeError("曲线段数不一致。")

    rows: List[Dict[str, int]] = []

    for seg, (la, lb) in enumerate(zip(curvesA, curvesB)):
        A_nodes, A_par = curve_nodes_sorted(la)
        B_nodes, B_par = curve_nodes_sorted(lb)

        if reverse_B:
            B_nodes = list(reversed(B_nodes))
            B_par = list(reversed(B_par))

        if len(A_nodes) != len(B_nodes):
            raise RuntimeError(
                f"第 {pair_index} 对，第 {seg} 段：A/B 节点数不一致 A={len(A_nodes)} B={len(B_nodes)}"
            )

        # 参数一致性检查（Transfinite + 同样点数时，一般可做到完全一致；tol_param 默认 0）
        for i in range(len(A_par)):
            if abs(A_par[i] - B_par[i]) > tol_param:
                raise RuntimeError(
                    f"第 {pair_index} 对，第 {seg} 段，第 {i} 个：参数不一致 "
                    f"A_par={A_par[i]} B_par={B_par[i]} (tol={tol_param})"
                )

        for i, (na, nb) in enumerate(zip(A_nodes, B_nodes)):
            rows.append({"pair": pair_index, "seg": seg, "i": i, "nodeA": na, "nodeB": nb})

    return rows


def main():
    gmsh.initialize()
    gmsh.model.add("portal_strict_match")

    # ==============
    # 1) 外域矩形
    # ==============
    Lx, Ly = 10.0, 6.0
    rect = gmsh.model.occ.addRectangle(-Lx / 2, -Ly / 2, 0.0, Lx, Ly, tag=1)

    # ==============
    # 2) 定义多对传送门（每对两条等长线段）
    # ==============
    portal_pairs: List[PortalPair] = [
        ((-2.0, -1.0, 0.0), (-2.0,  1.0, 0.0), ( 2.0, -1.0, 0.0), ( 2.0,  1.0, 0.0)),
        ((-1.0,  2.0, 0.0), ( 1.0,  2.0, 0.0), (-1.0, -2.0, 0.0), ( 1.0, -2.0, 0.0)),
    ]

    # 你可以在门上加入“必须出现”的关键点（按弧长参数 t）
    # 这些点不仅一定出现，还会参与“严格对应”的序列。
    t_breaks = [0.2, 0.55, 0.8]

    # 每一小段上再细分的点数（m+1）
    # - 若你要“门上除了关键点不允许任何额外点”，把 points_per_segment=2
    # - 若你要门上也有足够自由度用于耦合，设大一些，比如 11/21/41...
    points_per_segment = 11  # = m+1

    # 离散分布模式：
    # - 严格等间距：("Progression", 1.0)
    # - 两端更细但完全对应：("Bump", 4.0)  # coef 越大两端越细
    mode = "Progression"
    coef = 1.0

    # 是否需要把 B 门顺序反过来对应（用于“从另一侧穿出”的定义）
    reverse_B = False

    # ==============
    # 3) 创建并切分所有传送门
    # ==============
    portalA_curves: List[List[int]] = []
    portalB_curves: List[List[int]] = []
    all_portal_curves: List[int] = []

    for k, (A0, A1, B0, B1) in enumerate(portal_pairs):
        LA, LB = seg_len(A0, A1), seg_len(B0, B1)
        if abs(LA - LB) > 1e-12:
            raise ValueError(f"第 {k} 对传送门两侧长度不等，无法保证对应间距一致。LA={LA}, LB={LB}")

        # 给每对分配一段不冲突的 tag 区间（你也可以按自己习惯改）
        # 点 tag: 10000 + k*1000 + ...
        # 线 tag: 20000 + k*1000 + ...
        curvesA, _ = add_portal_as_polyline(
            A0, A1, t_breaks,
            point_tag_base=100000 + k * 1000,
            line_tag_base=200000 + k * 1000
        )
        curvesB, _ = add_portal_as_polyline(
            B0, B1, t_breaks,
            point_tag_base=110000 + k * 1000,
            line_tag_base=210000 + k * 1000
        )

        portalA_curves.append(curvesA)
        portalB_curves.append(curvesB)
        all_portal_curves.extend(curvesA + curvesB)

    gmsh.model.occ.synchronize()

    # ==============
    # 4) 嵌入到面网格，且逐段设置完全一致的 transfinite
    # ==============
    gmsh.model.mesh.embed(1, all_portal_curves, 2, rect)

    for k in range(len(portal_pairs)):
        set_identical_transfinite_for_pair(
            portalA_curves[k], portalB_curves[k],
            points_per_segment=points_per_segment,
            mode=mode,
            coef=coef
        )

    # （可选）对外边界或域内网格再设置一些尺寸场来加密“面”，不会改变门上的节点序列
    # 你如果需要，我可以把“门附近带状区域加密”的 field 也加上。

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    # ==============
    # 5) 生成网格
    # ==============
    gmsh.model.mesh.generate(2)
    gmsh.write("portal_strict_match.msh")

    # ==============
    # 6) 严格校验 + 导出映射
    # ==============
    all_rows: List[Dict[str, int]] = []
    for k in range(len(portal_pairs)):
        rows = build_mapping_and_validate(
            portalA_curves[k], portalB_curves[k],
            reverse_B=reverse_B,
            pair_index=k,
            tol_param=0.0  # 你说“不能有任何差异”，这里就设 0
        )
        all_rows.extend(rows)

    with open("portal_mapping.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "seg", "i", "nodeA", "nodeB"])
        w.writeheader()
        w.writerows(all_rows)

    gmsh.finalize()
    print("Wrote: portal_strict_match.msh")
    print("Wrote: portal_mapping.csv")
    print("Strict validation: PASSED")


if __name__ == "__main__":
    main()