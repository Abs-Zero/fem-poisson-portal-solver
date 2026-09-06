from portalsolver import PortalPoissonSolver, np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps
import os


# fmt:off
class CaseRun(PortalPoissonSolver):
    def __init__(
        self,
        portals, domain_shape, domain_size, mesh_size_regular, mesh_size_min,
        source_func, source_boundary, fig_name,
        switch_portal = True,
        show_progress = True,
        solve_method = "cg"
    ):
        super().__init__(
            portals=portals,
            domain_side_length=domain_size,
            mesh_size_regular=mesh_size_regular,
            mesh_size_min=mesh_size_min,
            switch_portal=switch_portal,
            show_progress=show_progress
        )

        self.generate_domain_mesh(domain_shape=domain_shape)
        self.solve_poisson(source_func, solve_method=solve_method)
        self.plot_solution(
            source_boundary=source_boundary,
            draw_contours=True,
            plot_portal=True,
            fig_name=fig_name
        )


def plot_sphere(f, fig_name="sphere.svg"):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm

    err = 1e-6
    u = np.linspace(np.pi/12+err, 11*np.pi/12-err, 200)
    v = np.linspace(-np.pi+err, np.pi-err, 300)
    U, V = np.meshgrid(u, v, indexing='ij')

    X = np.sin(U) * np.cos(V)
    Y = np.sin(U) * np.sin(V)
    Z = np.cos(U)

    F = np.empty(U.shape)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            if str(f(U[i, j], V[i, j])) == "nan":
                print(U[i, j], V[i, j])
            F[i, j] = f(U[i, j], V[i, j])
    cmap = colormaps["turbo"]
    norm = Normalize(vmin=np.nanmin(F), vmax=np.nanmax(F))
    C = cmap(norm(F))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, facecolors=C, linewidth=0, shade=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1,1,1))
    ax.axis('off')
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    fig.colorbar(mappable, ax=ax)
    
    os.makedirs("./output", exist_ok=True)
    plt.savefig(os.path.join("./output", fig_name))
    plt.show()


p1 = [[(1, 0, 2, 0), (-1, 0, 2, 0), False]]
p2 = [[(1, 0, 2, 0), (-1, 0, 2, np.pi), False]]
p3 = [
    [(1, 0, 2, 0), (-1, 0, 2, np.pi / 3), False],
    [(0, 2, 2, np.pi / 2), (0, -2, 2, np.pi / 3), False],
]
p4 = [[(1, 1, 3, np.pi / 2), (-1, -1, 3, -np.pi / 2), False]]
p5 = [
    [(1, 2, 2, 0), (-0.5, 1, 2, np.pi/6), True],
    [(1, -2, 2, np.pi/2), (-0.5, -1, 2, 0), True],
    [(0.4, 0, 2, 0), (4, 0, 2, 0), False],
]
p6 = [[(-4, 0, 9.99, 0), (4, 0, 9.99, 0), False]]
p7 = [
    [(-2, 0, 4, 0), (2, 0, 4, 0), True],
    [(0, 2.001, 4, -np.pi / 2), (0, -2.001, 4, -np.pi / 2), True],
]
p8 = [
    [(-2, 0, 4, 0), (2, 0, 4, 0), False],
    [(0, 2.001, 4, -np.pi / 2), (0, -2.001, 4, -np.pi / 2), True],
]
p9 = [[(-np.pi, 0, 4, 0), (np.pi, 0, 4, 0), False]]


def s_cir(x, y):
    G = 100
    R = 0.3
    dx, dy = x - 0, y - 0
    r = np.sqrt((dx) ** 2 + (dy) ** 2)
    res = 0
    if r < R:
        res = G
    return res

def s_rtg(x, y):
    C, H, L, D, G = 0, 6, 18, 5, 100
    res = 0
    xl, xr = C - L / 2, C + L / 2
    if xl <= x <= xr:
        if H - D <= y <= H:
            res = G
        if -H <= y <= -H + D:
            res = -G
    return res


def s_two_cir(x, y):
    G = 100
    R = 0.7
    dx, dy = x - 0.5, y - 0.5
    r = np.sqrt((dx) ** 2 + (dy) ** 2)
    res = 0
    if r < R:
        res = G
    dx, dy = x + 0.5, y + 0.5
    r = np.sqrt((dx) ** 2 + (dy) ** 2)
    if r < R:
        res = -G
    return res

H_sphere_xi = -np.log(np.tan(np.pi / 24))
def s_sphere_rtg(xi, eta):
    eta_min = np.log(np.tan(5 * np.pi / 24))
    eta_max = np.log(np.tan(7 * np.pi / 24))

    res = 0
    if eta_min < eta < eta_max:
        res = 100
    exp_eta = np.exp(eta)
    res *= 2 * exp_eta / (1 + exp_eta**2)

    return res



CaseRun(p1, "circle", 5, 0.1, 0.05, s_cir, [], "圆形边界，中心源，平行传送门.svg")
CaseRun(p2, "circle", 5, 0.1, 0.05, s_cir, [], "圆形边界，中心源，平行传送门，等效无门.svg")
CaseRun(p3, "circle", 5, 0.1, 0.05, s_cir, [], "圆形边界，中心源，2对门.svg")
CaseRun(p4, "rectangle", 10, 0.2, 0.1, s_rtg, [], "方形边界，平行条形场源，1对门.svg")
CaseRun(p5, "circle", 5, 0.1, 0.05, s_cir, [], "圆形边界，中心源，3对门.svg")
CaseRun(p6, "rectangle", 5, 0.1, 0.05, s_cir, [], "圆柱面热稳态模拟.svg")
CaseRun(p7, "rectangle", 5, 0.1, 0.05, s_two_cir, [], "环面稳态场模拟.svg")
CaseRun(p8, "rectangle", 5, 0.1, 0.05, s_two_cir, [], "克莱因瓶表面稳态场模拟.svg")
case9 = CaseRun(p9, "rectangle", (np.pi+0.001,H_sphere_xi), 0.1, 0.05, s_sphere_rtg, [], "球面热稳态模拟.svg")
plot_sphere(lambda x, y: case9.evaluate(y, np.log(np.tan(x/2))), fig_name="球面热稳态还原")

