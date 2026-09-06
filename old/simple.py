import gmsh
import numpy as np

gmsh.initialize()
gmsh.model.add("circle_mesh")

mesh_size = 0.1

circle = gmsh.model.occ.addCircle(0, 0, 0, 1)
gmsh.model.occ.addCurveLoop([circle], 1)
gmsh.model.occ.addPlaneSurface([1], 1)
gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
gmsh.model.mesh.generate(2)

node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
Nodes = np.array(node_coords).reshape(-1, 3)[:, :2]

elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(2, 1)
tri_node_tags = node_tags[0]

Elements = np.array(tri_node_tags, dtype=int).reshape(-1, 3) - 1

print("="*50)
print(f"总节点数: {len(Nodes)}")
print("Nodes (节点坐标 x,y):")
print(Nodes[:5])
print("="*50)
print(f"总三角形单元数: {len(Elements)}")
print("Elements (3个节点编号):")
print(Elements[:5])
print("="*50)

gmsh.fltk.run()
gmsh.write("circle_mesh.msh")

gmsh.finalize()