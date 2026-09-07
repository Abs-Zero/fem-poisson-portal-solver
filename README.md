# fem-poisson-portal-solver
实现了简单传送门作用下二维泊松方程的有限元求解

基于 **P1 Lagrange 有限元（Galerkin）** 求解带特殊拓扑连通（"传送门"）的二维 Poisson 方程 $-Δu=f$。通过网格生成与节点重连算法处理传送门对应的边界映射，在复杂连通域上求解场分布。

> **数学模型、弱形式推导与算法细节见本仓库 `论文.pdf`。**

## 仓库内容

| 文件 | 说明 |
|------|------|
| `portalsolver.py` | 核心求解器：有限元组装、传送门节点重连 |
| `mesh_express.py` | 绘制网格示意图 |
| `solve_express.py` | 求解演示 |
| `stability_express.py` | 收敛性与稳定性验证（网格加密、误差阶测试） |
| `portal_ploter.py` | 结果可视化 |
| `论文.pdf` | 完整推导与报告：连续模型、弱形式、离散格式、数值实验、收敛性分析 |
| `output/` | 数值实验结果图 |
| `old/` | 已弃用的历史代码 |

## 运行

依赖：Python 3.x、numpy、scipy、matplotlib、gmsh

```bash
python mesh_express.py      # 网格结构演示
python solve_express.py     # 求解演示
python portal_ploter.py     # 绘制传送门
python stability_express.py # 收敛性验证
