import numpy as np
import open3d as o3d
from matplotlib.path import Path

def load_point_cloud(file_path):
    """
    读取点云文件
    
    Args:
        file_path: 点云文件路径
        
    Returns:
        pcd: 读取的点云对象
    """
    print(f"正在读取点云文件: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"点云读取完成，共有 {len(pcd.points)} 个点")
    return pcd

def extract_obstacles(pcd, z_min=-1.0, z_max=0.6):
    """
    从点云中提取障碍物
    
    Args:
        pcd: 输入点云对象
        z_min: 最小高度阈值
        z_max: 最大高度阈值
        
    Returns:
        extracted_pcd: 提取的障碍物点云
    """
    points_input = np.asarray(pcd.points)
    
    # 创建基于z范围的掩码
    mask = (points_input[:, 2] >= z_min) & (points_input[:, 2] <= z_max)
    
    # 过滤点
    obstacle_points_extracted = points_input[mask]
    
    # 为提取的障碍物创建新的点云对象
    extracted_pcd = o3d.geometry.PointCloud()
    extracted_pcd.points = o3d.utility.Vector3dVector(obstacle_points_extracted)
    
    # 如果输入pcd有颜色信息，并且颜色数组与点数组长度匹配，则过滤并分配颜色
    if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
        colors_input = np.asarray(pcd.colors)
        if colors_input.shape[0] == points_input.shape[0]:  # 确保原始点和颜色数量一致
            obstacle_colors_extracted = colors_input[mask]
            extracted_pcd.colors = o3d.utility.Vector3dVector(obstacle_colors_extracted)
        else:
            print(f"警告: 输入点云的颜色数组长度 ({colors_input.shape[0]}) "
                  f"与点数组长度 ({points_input.shape[0]}) 不匹配。 "
                  "提取的障碍物将不包含原始颜色。")

    print(f"已提取障碍物点云，共 {len(obstacle_points_extracted)} 个点")
    return extracted_pcd

def create_goal_point_cloud(goal_point, radius=0.5, num_points=100):
    """
    创建表示目标的球形点云
    
    Args:
        goal_point: 目标点坐标 (x, y, z)
        radius: 球体半径
        num_points: 球体上的点数量
        
    Returns:
        pcd_sphere: 球形点云对象
    """
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    pcd_sphere = mesh_sphere.sample_points_uniformly(number_of_points=num_points)
    
    points = np.asarray(pcd_sphere.points) + goal_point
    pcd_sphere.points = o3d.utility.Vector3dVector(points)
    return pcd_sphere

def create_pool_obstacle(corner_points, z_range=(0, 1), density=0.1):
    """
    创建表示水池的障碍物点云
    
    Args:
        corner_points: 水池四个角点的坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        z_range: 水池的高度范围 (min_z, max_z)
        density: 生成点云的密度，值越小点越密集
        
    Returns:
        water_pool_pcd: 水池的点云对象
    """
    # 将角点转换为numpy数组
    corners = np.array(corner_points)
    
    # 计算边界
    min_x = np.min(corners[:, 0])
    max_x = np.max(corners[:, 0])
    min_y = np.min(corners[:, 1])
    max_y = np.max(corners[:, 1])
    
    # 根据密度创建均匀分布的网格点
    x_coords = np.arange(min_x, max_x, density)
    y_coords = np.arange(min_y, max_y, density)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # 将网格点转换为坐标列表
    grid_points = np.column_stack((xx.flatten(), yy.flatten()))
    
    # 使用多边形包含测试来筛选出水池内的点
    polygon = Path(corners)
    mask = polygon.contains_points(grid_points)
    pool_points = grid_points[mask]
    
    # 为每个点添加不同的z坐标，创建3D点云
    z_min, z_max = z_range
    z_count = max(1, int((z_max - z_min) / density))
    z_values = np.linspace(z_min, z_max, z_count)
    
    # 为每个xy点生成多个z高度的点
    all_points = []
    for point in pool_points:
        for z in z_values:
            all_points.append([point[0], point[1], z])
    
    all_points = np.array(all_points)
    
    # 创建Open3D点云对象
    water_pool_pcd = o3d.geometry.PointCloud()
    water_pool_pcd.points = o3d.utility.Vector3dVector(all_points)
    
    # 设置蓝色
    colors = np.zeros((len(all_points), 3))
    colors[:, 2] = 0.8  # 蓝色 (R=0, G=0, B=0.8)
    water_pool_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"已创建水池障碍物点云，共 {len(all_points)} 个点")
    return water_pool_pcd

def create_square_obstacle(center_x, center_y, width, z_min=0.0, z_max=0.6, density=0.05):
    """
    创建表示方形障碍物的点云
    
    Args:
        center_x: 障碍物中心X坐标
        center_y: 障碍物中心Y坐标  
        width: 障碍物宽度（正方形边长）
        z_min: 障碍物的最小高度
        z_max: 障碍物的最大高度
        density: 生成点云的密度，值越小点越密集
        
    Returns:
        obstacle_pcd: 方形障碍物的点云对象
    """
    # 计算边界
    half_width = width / 2.0
    min_x = center_x - half_width
    max_x = center_x + half_width
    min_y = center_y - half_width
    max_y = center_y + half_width
    
    # 根据密度创建均匀分布的网格点
    x_coords = np.arange(min_x, max_x, density)
    y_coords = np.arange(min_y, max_y, density)
    z_coords = np.arange(z_min, z_max, density)
    
    # 创建3D网格
    all_points = []
    for x in x_coords:
        for y in y_coords:
            for z in z_coords:
                all_points.append([x, y, z])
    
    all_points = np.array(all_points)
    
    # 创建Open3D点云对象
    obstacle_pcd = o3d.geometry.PointCloud()
    obstacle_pcd.points = o3d.utility.Vector3dVector(all_points)
    
    # 设置灰色
    colors = np.ones((len(all_points), 3)) * 0.5  # 灰色 (R=0.5, G=0.5, B=0.5)
    obstacle_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"已创建方形障碍物点云，中心: ({center_x:.1f}, {center_y:.1f})，宽度: {width:.1f}m，共 {len(all_points)} 个点")
    return obstacle_pcd

def remove_points_in_polygon(pcd, polygon_vertices, z_min=-float('inf'), z_max=float('inf')):
    """
    移除点云中位于指定多边形区域内的点
    
    Args:
        pcd: 输入的点云对象 (open3d PointCloud)
        polygon_vertices: 多边形顶点坐标列表 [(x1,y1), (x2,y2), ...]
        z_min: 可选的最小高度限制
        z_max: 可选的最大高度限制
        
    Returns:
        filtered_pcd: 移除了多边形区域内点的点云对象
    """
    # 转换为numpy数组
    points = np.asarray(pcd.points)
    
    # 创建多边形路径对象
    polygon = Path(polygon_vertices)
    
    # 检查每个点的xy坐标是否在多边形内
    inside_polygon = polygon.contains_points(points[:, :2])
    
    # 同时考虑z范围
    in_z_range = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    
    # 确定要移除的点和要保留的点
    to_remove = inside_polygon & in_z_range
    to_keep = ~to_remove
    
    # 保留不在多边形内的点
    filtered_points = points[to_keep]
    
    # 创建新的点云对象
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    # 如果原点云有颜色信息，也保留下来
    if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
        colors = np.asarray(pcd.colors)
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[to_keep])
    
    removed_count = np.sum(to_remove)
    print(f"已从点云中移除多边形区域内的 {removed_count} 个点 (总点数: {len(points)})")
    
    return filtered_pcd