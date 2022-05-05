import numpy as np
import open3d as o3d

# http://www.open3d.org/docs/0.12.0/tutorial/geometry/pointcloud.html

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("xxx.pcd")
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd])

print("Recompute the normal of the point cloud")
pcd.normals.clear()
pcd.estimate_normals(
    o3d.geometry.KDTreeSearchParamRadius(radius=0.1)
    )
o3d.visualization.draw_geometries([pcd])
