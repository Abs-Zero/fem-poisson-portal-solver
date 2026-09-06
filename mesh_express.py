from portalsolver import PortalPoissonSolver, np

portals = [
    [(3, 0, 5, 0), (-1, 0, 5, np.pi / 3), False],
]

def prime_mesh():  # 初始网格示意图
    solver = PortalPoissonSolver(
        [],
        domain_side_length=5,
        mesh_size_regular=3,
        mesh_size_min=3,
        switch_portal=False,
        show_progress=False,
    )

    solver.generate_domain_mesh(domain_shape="circle")

    solver.plot_mesh(
        show_ele_tags=True,
        plot_boundary_nodes=True,
        plot_portal_points=True,
        plot_portal=True,
        fig_name="初始网格示意图.svg",
    )

def portal_forced_mesh():  # 添加传送门后网格示意图
    solver = PortalPoissonSolver(
        portals,
        domain_side_length=5,
        mesh_size_regular=3,
        mesh_size_min=3,
        switch_portal=False,
        show_progress=False,
    )

    solver.generate_domain_mesh(domain_shape="circle")

    solver.plot_mesh(
        show_ele_tags=True,
        plot_boundary_nodes=True,
        plot_portal_points=True,
        plot_portal=True,
        fig_name="添加传送门后网格示意图.svg",
    )

def switched_mesh():  # 变换后网格示意图
    solver = PortalPoissonSolver(
        portals,
        domain_side_length=5,
        mesh_size_regular=3,
        mesh_size_min=3,
        switch_portal=True,
        show_progress=False,
    )

    solver.generate_domain_mesh(domain_shape="circle")

    solver.plot_mesh(
        show_ele_tags=True,
        plot_boundary_nodes=True,
        plot_portal_points=True,
        plot_portal=True,
        fig_name="变换后网格示意图.svg",
    )

def square_domain_mesh():  # 方形定义域下的网格
    solver = PortalPoissonSolver(
        portals,
        domain_side_length=5,
        mesh_size_regular=3,
        mesh_size_min=3,
        switch_portal=False,
        show_progress=False,
    )

    solver.generate_domain_mesh(domain_shape="rectangle")

    solver.plot_mesh(
        show_ele_tags=True,
        plot_boundary_nodes=True,
        plot_portal_points=True,
        plot_portal=True,
        fig_name="方形定义域网格.svg",
    )



prime_mesh()
portal_forced_mesh()
switched_mesh()
square_domain_mesh()