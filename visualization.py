import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import open3d as o3d
import matplotlib.patheffects as path_effects

def visualize_path(path2d, start_point, end_point, save_path=None, ax=None, color='b', label=None):
    """
    单个路径的可视化函数，可以指定绘制到特定的坐标轴对象上
    
    Args:
        path2d: 2D路径点列表
        start_point: 起点坐标
        end_point: 终点坐标
        save_path: 保存图像的路径
        ax: matplotlib轴对象
        color: 路径颜色
        label: 路径标签
    
    Returns:
        ax: matplotlib轴对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    x = [p[0] for p in path2d]
    y = [p[1] for p in path2d]
    
    # 绘制路径
    ax.plot(x, y, color=color, linewidth=2, label=label)
    
    # 绘制起点和终点
    ax.plot(0, 0, 'go', markersize=8)  # 起点始终为原点(相对坐标系)
    ax.plot(x[-1], y[-1], 'o', color=color, markersize=8)
    
    if save_path and ax is None:  # 只有在独立使用时才保存
        plt.grid(True)
        plt.title(f'路径：({start_point[0]:.1f}, {start_point[1]:.1f}) → ({end_point[0]:.1f}, {end_point[1]:.1f})')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.savefig(save_path)
        plt.close()
    
    return ax

def visualize_path_with_obstacles(path2d, start_point, end_point, obstacle_pcd=None, 
                                 save_path=None, ax=None, color='b', label=None, 
                                 z_min=-1.0, z_max=1):
    """
    增强的路径可视化函数，同时显示路径和障碍物点云投影
    
    Args:
        path2d: 2D路径点列表
        start_point: 起点坐标
        end_point: 终点坐标
        obstacle_pcd: 障碍物点云对象 (open3d PointCloud)
        save_path: 保存图像的路径
        ax: matplotlib轴对象
        color: 路径颜色
        label: 路径标签
        z_min: 障碍物点云的最小高度
        z_max: 障碍物点云的最大高度
    
    Returns:
        ax: matplotlib轴对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. 绘制障碍物点云投影（如果提供）
    if obstacle_pcd is not None:
        points = np.asarray(obstacle_pcd.points)
        
        # 按高度过滤点云
        mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        filtered_points = points[mask]
        
        # 获取颜色信息（如果有）
        if hasattr(obstacle_pcd, 'colors') and len(obstacle_pcd.colors) > 0:
            colors = np.asarray(obstacle_pcd.colors)[mask]
            # 绘制彩色点云投影
            ax.scatter(filtered_points[:, 0], filtered_points[:, 1], 
                      s=0.5, c=colors, alpha=0.5, label='Obstacles')
        else:
            # 没有颜色信息时使用红色
            ax.scatter(filtered_points[:, 0], filtered_points[:, 1], 
                      s=0.5, c='red', alpha=0.5, label='Obstacles')
        
        print(f"已绘制{len(filtered_points)}个障碍物点")
    
    # 2. 绘制路径（保持原有功能）
    x = [p[0] for p in path2d]
    y = [p[1] for p in path2d]
    
    # 绘制路径
    ax.plot(x, y, color=color, linewidth=2, label=label if label else 'Planned Path')
    
    # 绘制起点和终点
    ax.plot(0, 0, 'go', markersize=8, label='Start')  # 相对坐标系中的起点
    ax.plot(x[-1], y[-1], 'o', color=color, markersize=8, label='End')
    
    # 添加网格和图例
    ax.grid(True)
    ax.set_title(f'Path and Obstacles: ({start_point[0]:.1f}, {start_point[1]:.1f}) → ({end_point[0]:.1f}, {end_point[1]:.1f})')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='best')
    
    # 保存图像
    if save_path and ax is None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"路径与障碍物可视化已保存到 {save_path}")
        plt.close()
    
    return ax

def visualize_all_paths_with_pointcloud(all_paths_data, pcd, output_file, z_min=0.35, z_max=1.2):
    """
    可视化所有路径和障碍物点云投影（使用绝对坐标）
    
    Args:
        all_paths_data: 包含所有路径数据的列表
        pcd: 点云对象
        output_file: 输出图像的路径
        z_min: 障碍物点云的最小高度
        z_max: 障碍物点云的最大高度
    """
    # 创建图形
    dpi = 300
    fig, ax = plt.subplots(figsize=(20, 12), dpi=dpi)
    
    # 只提取障碍物区域的点云（z_min到z_max之间）
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    obstacle_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    
    # 只保留障碍物点云
    obstacle_points = points[obstacle_mask]
    obstacle_colors = colors[obstacle_mask]
    
    # 绘制障碍物点云投影，使用原始颜色
    ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], 
              s=0.5, c=obstacle_colors, alpha=0.6, label='Obstacles')
    
    print(f"已绘制 {len(obstacle_points)} 个障碍物点云")
    
    # 获取matplotlib颜色循环
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # 收集所有路径点用于后续确定坐标轴范围
    all_path_points = []

    shadow_effect = path_effects.withSimplePatchShadow(
        offset=(1, -1),
        shadow_rgbFace='black',
        alpha=0.8
    )

    # 绘制所有路径（使用绝对坐标）
    for i, path_data in enumerate(all_paths_data):
        # 获取路径数据
        segment = path_data['segment']
        start_point = path_data['start']
        end_point = path_data['end']

        # 相对路径转换回绝对路径
        rel_path2d = path_data['path2d']
        abs_path2d = [(p[0] + start_point[0], p[1] + start_point[1]) for p in rel_path2d]
        all_path_points.extend(abs_path2d)

        # 选择颜色
        color = color_cycle[i % len(color_cycle)]

        # 绘制路径
        label = f'Path {segment}'
        x = [p[0] for p in abs_path2d]
        y = [p[1] for p in abs_path2d]
        ax.plot(x, y, color=color, linewidth=2, label=label)

        # 绘制起点和终点
        ax.plot(start_point[0], start_point[1], 'o', color=color, markersize=8)
        ax.plot(end_point[0], end_point[1], 's', color=color, markersize=8)

        # 添加起点标注 with shadow
        ax.annotate(f'Start point {i+1}', (start_point[0], start_point[1]),
                    xytext=(10, -15), textcoords='offset points',
                    color=color, fontsize=20, fontweight='bold',
                    path_effects=[shadow_effect]) # MODIFIED: Applied new shadow effect

        if i == len(all_paths_data) - 1:  # 最后一个路径的终点
            # 添加终点标注 with shadow
            ax.annotate(f'End point {i+2}', (end_point[0], end_point[1]),
                        xytext=(10, -15), textcoords='offset points',
                        color=color, fontsize=20, fontweight='bold',
                        path_effects=[shadow_effect]) # MODIFIED: Applied new shadow effect


    # 设置图形样式
    ax.grid(True)
    ax.set_title('Path and Obstacles Projection')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='best')
    
    ax.set_aspect('equal')
    # 调整坐标轴范围，确保所有路径和障碍物点云都可见
    margin = 1.0  # 边缘留白
    
    # 获取所有点的坐标范围
    all_x = [p[0] for p in all_path_points] + list(obstacle_points[:, 0])
    all_y = [p[1] for p in all_path_points] + list(obstacle_points[:, 1])
    
    # 计算可视范围
    x_min = min(all_x) - margin
    x_max = max(all_x) + margin
    y_min = min(all_y) - margin
    y_max = max(all_y) + margin
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi)
    start_pixel = ax.transData.transform(np.array([[start_point[0], start_point[1]]]))
    end_pixel = ax.transData.transform(np.array([[end_point[0], end_point[1]]]))
    
    # 使用图形大小上下文打印实际位置
    fig_size = fig.get_size_inches() * dpi
    print(f"图像大小: {fig_size}")
    print(f"Path {segment} - Start point actual coordinates: {start_pixel[0]}")
    print(f"Path {segment} - End point actual coordinates: {end_pixel[0]}")
    print(f"所有路径和障碍物点云投影已保存到 {output_file}")
    plt.close(fig)  # 显式关闭图形对象

    mapping = {
        "start_point": start_pixel[0].tolist(),
        "end_point": end_pixel[0].tolist(),
    }

    return mapping

def add_path_to_pointcloud(pcd, all_paths_data, z_height=0.6, output_file=None, 
                           path_point_interval_along=0.1, 
                           marker_side_length=0.2, marker_point_spacing=0.05,
                           path_line_width=0.1, path_line_points_across=3,
                           element_diameter=0.1, element_subpoint_spacing=0.02):
    """
    将生成的所有路径添加到点云文件中，以固定高度显示。
    
    Args:
        pcd: 原始点云对象
        all_paths_data: 包含所有路径数据的列表
        z_height: 路径点的固定高度
        output_file: 输出的PLY文件路径
        path_point_interval_along: 沿路径方向的点之间的间隔
        marker_side_length: 起点/终点标记物的边长
        marker_point_spacing: 标记物内部点之间的间距
        path_line_width: 路径线的宽度
        path_line_points_across: 路径线宽度上的点数
        element_diameter: 每个点元素的直径
        element_subpoint_spacing: 每个点元素内部子点的间距
    
    Returns:
        combined_pcd: 合并后的点云对象
    """
    original_points = np.asarray(pcd.points)
    original_colors = np.asarray(pcd.colors) if hasattr(pcd, 'colors') and len(pcd.colors) > 0 else \
                     np.ones((len(original_points), 3)) * 0.7
    
    path_line_color = np.array([1.0, 1.0, 0.0])  # 黄色
    segment_start_color = np.array([0.0, 1.0, 0.0])  # 绿色
    segment_end_color = np.array([1.0, 0.0, 0.0])  # 红色
    
    all_final_subpoints = []
    all_final_subcolors = []

    def add_volumetric_element(center_xyz_arr, color_val):
        center_x, center_y, center_z = center_xyz_arr
        half_diam = element_diameter / 2.0
        # 确保即使直径很小，也至少有一个子点
        num_steps_per_dim = max(1, int(element_diameter / element_subpoint_spacing))
        
        # 如果直径太小，只添加中心点
        if element_diameter < 1e-4 or num_steps_per_dim == 0:
            all_final_subpoints.append([center_x, center_y, center_z])
            all_final_subcolors.append(color_val)
            return
            
        offsets = np.linspace(-half_diam, half_diam, num_steps_per_dim + 1)

        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    all_final_subpoints.append([center_x + dx, center_y + dy, center_z + dz])
                    all_final_subcolors.append(color_val)

    for i, path_data in enumerate(all_paths_data):
        rel_path2d = path_data['path2d']
        abs_segment_start_xy = (path_data['start'][0], path_data['start'][1])
        abs_segment_end_xy = (path_data['end'][0], path_data['end'][1])
        
        # 1. 添加起点标记
        if marker_side_length == 0 or marker_point_spacing == 0:  # 如果边长或间距为零，则只有一个元素
            marker_center_offsets_x = [0.0]
            marker_center_offsets_y = [0.0]
        else:
            num_marker_elements_per_side_x = max(1, int(marker_side_length / marker_point_spacing) + 1)
            num_marker_elements_per_side_y = max(1, int(marker_side_length / marker_point_spacing) + 1)
            marker_center_offsets_x = np.linspace(-marker_side_length / 2, marker_side_length / 2, num_marker_elements_per_side_x)
            marker_center_offsets_y = np.linspace(-marker_side_length / 2, marker_side_length / 2, num_marker_elements_per_side_y)

        for dx_offset in marker_center_offsets_x:
            for dy_offset in marker_center_offsets_y:
                element_center = [abs_segment_start_xy[0] + dx_offset, abs_segment_start_xy[1] + dy_offset, z_height]
                add_volumetric_element(element_center, segment_start_color)
        
        # 2. 添加路径线
        if rel_path2d and len(rel_path2d) > 1:
            for j in range(len(rel_path2d) - 1):
                p1_rel, p2_rel = rel_path2d[j], rel_path2d[j+1]
                abs_p1_x, abs_p1_y = p1_rel[0] + abs_segment_start_xy[0], p1_rel[1] + abs_segment_start_xy[1]
                abs_p2_x, abs_p2_y = p2_rel[0] + abs_segment_start_xy[0], p2_rel[1] + abs_segment_start_xy[1]
                
                dist = np.sqrt((abs_p2_x - abs_p1_x)**2 + (abs_p2_y - abs_p1_y)**2)
                num_centerline_points = max(2, int(dist / path_point_interval_along) + 1) if path_point_interval_along > 1e-6 else 2
                
                segment_dx, segment_dy = abs_p2_x - abs_p1_x, abs_p2_y - abs_p1_y
                seg_len = np.sqrt(segment_dx**2 + segment_dy**2)
                perp_dx, perp_dy = (-segment_dy / seg_len, segment_dx / seg_len) if seg_len > 1e-6 else (0, 1)
                
                centerline_t_values = np.linspace(0, 1, num_centerline_points)
                for t in centerline_t_values:
                    centerline_x = abs_p1_x * (1 - t) + abs_p2_x * t
                    centerline_y = abs_p1_y * (1 - t) + abs_p2_y * t
                    
                    # path_line_points_across必须至少为1
                    actual_points_across = max(1, path_line_points_across)
                    width_strip_offsets = np.linspace(-path_line_width / 2, path_line_width / 2, actual_points_across)
                    if path_line_width == 0:  # 如果宽度为零，则只有一行元素
                        width_strip_offsets = [0.0]
                        
                    for strip_offset in width_strip_offsets:
                        element_center = [centerline_x + perp_dx * strip_offset, 
                                          centerline_y + perp_dy * strip_offset, 
                                          z_height]
                        add_volumetric_element(element_center, path_line_color)

        # 3. 添加终点标记
        for dx_offset in marker_center_offsets_x:
            for dy_offset in marker_center_offsets_y:
                element_center = [abs_segment_end_xy[0] + dx_offset, abs_segment_end_xy[1] + dy_offset, z_height]
                add_volumetric_element(element_center, segment_end_color)
            
    if not all_final_subpoints:
        print("警告: 未生成任何路径子点。")
        return pcd

    print(f"已生成 {len(all_final_subpoints)} 个路径子点")
    
    np_new_subpoints = np.array(all_final_subpoints)
    np_new_subcolors = np.array(all_final_subcolors)

    if np_new_subpoints.size == 0:
        return pcd
    if np_new_subpoints.ndim == 1:  # 如果子点是[x,y,z]，不应该发生
         np_new_subpoints = np_new_subpoints.reshape(-1, 3)
         np_new_subcolors = np_new_subcolors.reshape(-1, 3)

    combined_points = np.vstack([original_points, np_new_subpoints])
    combined_colors = np.vstack([original_colors, np_new_subcolors])
    
    # 创建新点云对象
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    # 保存为PLY文件
    if output_file:
        o3d.io.write_point_cloud(output_file, combined_pcd)
        print(f"已将轨迹添加到点云并保存为 {output_file}")
    
    return combined_pcd
