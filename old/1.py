import gmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

Point3 = Tuple[float, float, float]
PortalPair = Tuple[Point3, Point3, Point3, Point3]  # A0,A1,B0,B1


def add_portal_pair_occ(A0: Point3, A1: Point3, B0: Point3, B1: Point3) -> Tuple[List[int], List[int]]:
    pA0 = gmsh.model.occ.addPoint(A0[0], A0[1], A0[2])
    pA1 = gmsh.model.occ.addPoint(A1[0], A1[1], A1[2])
    pB0 = gmsh.model.occ.addPoint(B0[0], B0[1], B0[2])
    pB1 = gmsh.model.occ.addPoint(B1[0], B1[1], B1[2])

    lA = gmsh.model.occ.addLine(pA0, pA1)
    lB = gmsh.model.occ.addLine(pB0, pB1)

    return [lA, lB], [pA0, pA1, pB0, pB1]


def plot_gmsh_mesh_2d(msh_path: str, out_image_path: str, dpi: int = 300) -> None:
    """
    读取 gmsh .msh，用 matplotlib 绘制 2D 三角网格并保存。
    支持 msh 里含 triangle / triangle6 等三角单元；优先画三角形的边。
    """
    m = meshio.read(msh_path)

    pts = m.points[:, :2]  # 只取 x,y
    tri = None

    # meshio 里三角单元可能叫 "triangle" 或 "triangle6"
    for cell_block in m.cells:
        if cell_block.type == "triangle":
            tri = cell_block.data
            break

    if tri is None:
        # 如果只有二次三角形 triangle6，取前三个顶点做边框绘制（简化）
        for cell_block in m.cells:
            if cell_block.type == "triangle6":
                tri = cell_block.data[:, :3]
                break

    if tri is None:
        raise RuntimeError("未在 msh 中找到 triangle/triangle6 单元，无法绘制 2D 网格。")

    fig, ax = plt.subplots()
    ax.triplot(pts[:, 0], pts[:, 1], tri, linewidth=0.2)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Gmsh mesh")
    ax.margins(0.02)

    fig.savefig(out_image_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    gmsh.initialize()
    gmsh.model.add("portal_rect")

    # ----------------------------
    # 1) 大矩形域：中心原点
    # ----------------------------
    Lx, Ly = 10.0, 6.0
    x0, y0 = -Lx / 2.0, -Ly / 2.0
    rect_surf = gmsh.model.occ.addRectangle(x0, y0, 0.0, Lx, Ly)

    # ----------------------------
    # 2) 多对传送门线段
    # ----------------------------
    portal_pairs: List[PortalPair] = [
        ((-2.0, -1.0, 0.0), (-2.0,  1.0, 0.0), ( 2.0, -1.0, 0.0), ( 2.0,  1.0, 0.0)),
        ((-1.0,  2.0, 0.0), ( 1.0,  2.0, 0.0), (-1.0, -2.0, 0.0), ( 1.0, -2.0, 0.0)),
    ]

    portal_line_tags: List[int] = []
    portal_point_tags: List[int] = []

    for (A0, A1, B0, B1) in portal_pairs:
        ls, ps = add_portal_pair_occ(A0, A1, B0, B1)
        portal_line_tags.extend(ls)
        portal_point_tags.extend(ps)

    gmsh.model.occ.synchronize()

    # ----------------------------
    # 3) embed：让内部线段成为网格边
    # ----------------------------
    gmsh.model.mesh.embed(1, portal_line_tags, 2, rect_surf)

    # ----------------------------
    # 4) 网格尺寸场：线附近细、端点更细
    # ----------------------------
    h_far = 0.5
    h_near_line = 0.08
    h_near_pt = 0.03

    d_line_min = 0.10
    d_line_max = 0.60

    d_pt_min = 0.06
    d_pt_max = 0.25

    f_dist_line = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist_line, "CurvesList", portal_line_tags)
    gmsh.model.mesh.field.setNumber(f_dist_line, "Sampling", 100)

    f_th_line = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_th_line, "InField", f_dist_line)
    gmsh.model.mesh.field.setNumber(f_th_line, "SizeMin", h_near_line)
    gmsh.model.mesh.field.setNumber(f_th_line, "SizeMax", h_far)
    gmsh.model.mesh.field.setNumber(f_th_line, "DistMin", d_line_min)
    gmsh.model.mesh.field.setNumber(f_th_line, "DistMax", d_line_max)

    f_dist_pt = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist_pt, "PointsList", portal_point_tags)
    gmsh.model.mesh.field.setNumber(f_dist_pt, "Sampling", 100)

    f_th_pt = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_th_pt, "InField", f_dist_pt)
    gmsh.model.mesh.field.setNumber(f_th_pt, "SizeMin", h_near_pt)
    gmsh.model.mesh.field.setNumber(f_th_pt, "SizeMax", h_near_line)
    gmsh.model.mesh.field.setNumber(f_th_pt, "DistMin", d_pt_min)
    gmsh.model.mesh.field.setNumber(f_th_pt, "DistMax", d_pt_max)

    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_th_line, f_th_pt])
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    # 常用 2D 算法与优化
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    # ----------------------------
    # 5) Physical groups（可选但建议）
    # ----------------------------
    pg_portals = gmsh.model.addPhysicalGroup(1, portal_line_tags)
    gmsh.model.setPhysicalName(1, pg_portals, "portals")

    boundary_curves = gmsh.model.getBoundary([(2, rect_surf)], oriented=False, recursive=False)
    outer_curve_tags = [tag for (dim, tag) in boundary_curves if dim == 1]
    pg_outer = gmsh.model.addPhysicalGroup(1, outer_curve_tags)
    gmsh.model.setPhysicalName(1, pg_outer, "outer_boundary")

    pg_domain = gmsh.model.addPhysicalGroup(2, [rect_surf])
    gmsh.model.setPhysicalName(2, pg_domain, "domain")

    # ----------------------------
    # 6) 生成网格并导出 msh
    # ----------------------------
    gmsh.model.mesh.generate(2)
    msh_path = "portal_rect.msh"
    gmsh.write(msh_path)

    gmsh.finalize()
    print(f"Wrote: {msh_path}")

    # ----------------------------
    # 7) Python 绘图并保存
    # ----------------------------
    img_path = "portal_rect_mesh.png"
    plot_gmsh_mesh_2d(msh_path, img_path, dpi=3000)
    print(f"Saved mesh image: {img_path}")


if __name__ == "__main__":
    main()