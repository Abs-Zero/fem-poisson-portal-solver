import numpy as np
import matplotlib.pyplot as plt
from portalsolver import PortalPoissonSolver


pi = np.pi
def exact_solution(x, y):  # 解析解
    return np.sin(pi*(x+1)/2) * np.sin(pi*(y+1)/2)

def source_func(x, y):  # 泊松方程右端源项
    return (pi**2 / 2) * np.sin(pi*(x+1)/2) * np.sin(pi*(y+1)/2)

def plot_error():
    solver = PortalPoissonSolver(
        portals=[],
        domain_side_length=1.0,
        mesh_size_regular=0.01,
        mesh_size_min=0.01,
        switch_portal=True,
        show_progress=True
    )

    solver.generate_domain_mesh(domain_shape="rectangle")
    solver.solve_poisson(func_domain=source_func)
    
    print("绘制数值解、解析解对比图...")
    N = 100
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    numerical = np.zeros_like(X)
    exact = np.zeros_like(X)
    error = np.zeros_like(X)

    for i in range(N):
        for j in range(N):
            xi = X[i, j]
            yi = Y[i, j]
            numerical[i, j] = solver.evaluate(xi, yi)
            exact[i, j] = exact_solution(xi, yi)
            error[i, j] = np.abs(numerical[i, j] - exact[i, j])

    numerical = np.nan_to_num(numerical, nan=0)
    exact = np.nan_to_num(exact, nan=0)
    error = np.nan_to_num(error, nan=0)

    l2_error = np.sqrt(np.mean(error ** 2))
    max_error = np.max(error)

    print(f"L2 平均误差：{l2_error:.6f}")
    print(f"最大绝对误差：{max_error:.6f}")

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = "turbo"

    im1 = axes[0].pcolormesh(X, Y, numerical, cmap=cmap, shading='gouraud')
    axes[0].set_title("Numerical Solution(h=0.02)", fontsize=12, fontweight='bold')
    axes[0].axis("equal")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].pcolormesh(X, Y, exact, cmap=cmap, shading='gouraud')
    axes[1].set_title("Exact Solution", fontsize=12, fontweight='bold')
    axes[1].axis("equal")
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].pcolormesh(X, Y, error, cmap="coolwarm", shading='gouraud')
    axes[2].set_title(f"Absolute Error", fontsize=12, fontweight='bold')
    axes[2].axis("equal")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig("./output/稳定性分析-数值解与解析解对比.png", dpi=300)
    plt.show()

def calculate_error(mesh_size, x=0, y=0):
    solver = PortalPoissonSolver(
        portals=[],
        domain_side_length=1.0,
        mesh_size_regular=mesh_size,
        mesh_size_min=mesh_size,
        switch_portal=True,
        show_progress=False
    )

    solver.generate_domain_mesh(domain_shape="rectangle")
    solver.solve_poisson(func_domain=source_func)
    
    numerical = solver.evaluate(x, y)
    exact = exact_solution(x, y)
    error = np.abs(numerical - exact)

    return error

def plot_error_h(hl=0.02,hr=0.3):
    # hs = np.logspace(np.log10(hl), np.log10(hr), 20)
    hs = np.linspace(hl, hr, 100)
    errors = np.zeros_like(hs)

    for i, h in enumerate(hs):
        errors[i] = calculate_error(h)

    plt.figure(figsize=(8, 5))
    plt.plot(hs, errors, linewidth=2)

    plt.xlabel('Mesh Size $h$', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.title('Convergence Test: Error & Mesh Size', fontsize=14, pad=10)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./output/收敛性测试-误差-网格尺寸图.svg")
    plt.show()


if __name__ == "__main__":
    plot_error()
    plot_error_h()
