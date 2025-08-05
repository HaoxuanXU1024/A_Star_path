import os
import json
import time
import argparse
import numpy as np
import open3d as o3d
from utils import (
    load_point_cloud, 
    extract_obstacles, 
    create_goal_point_cloud,
    create_pool_obstacle,
    create_square_obstacle,
    remove_points_in_polygon
)
from astar import (
    a_star_optimized, 
    adjust_path,
    generate_robot_commands
)
from visualization import (
    visualize_path, 
    visualize_path_with_obstacles,
    visualize_all_paths_with_pointcloud,
    add_path_to_pointcloud
)

from parsers import (
    parse_arguments, 
    parse_polygon, 
    parse_waypoints,
    parse_obstacles,
)

def main(args):
    """主函数"""
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取点云文件
    pcd = load_point_cloud(args.input)
    
    # 如果指定了要移除的多边形区域
    if args.remove_polygon:
        default_polygon_str = "-19,-10.5;-19,-6.1;-14.2,-5.7;-14.2,-10.2"
        print(f"移除默认多边形区域: {default_polygon_str}")
        polygon = parse_polygon(default_polygon_str)
        pcd = remove_points_in_polygon(pcd, polygon)

    # 提取障碍物
    obstacle_pcd = extract_obstacles(pcd, z_min=args.obstacle_z_min, z_max=args.obstacle_z_max)
    obstacles = [obstacle_pcd]
    
    # 添加额外的障碍物
    if args.add_obstacles:
        additional_obstacles = parse_obstacles(args.add_obstacles)
        for x, y, width in additional_obstacles:
            square_obstacle = create_square_obstacle(
                center_x=x, 
                center_y=y, 
                width=width,
                z_min=args.obstacle_z_min,
                z_max=args.obstacle_z_max,
                density=0.05
            )
            obstacles.append(square_obstacle)
            print(f"已添加方形障碍物: 中心({x:.1f}, {y:.1f})，宽度{width:.1f}m")
    
    # 解析路径点
    if args.waypoints:
        waypoints = parse_waypoints(args.waypoints)
    else:
        # 默认路径点示例
        waypoints = [
            (-6.8, -5.0, 1.0),
            (-15.5, -8.0, 1.0)
        ]
        print(f"使用默认路径点: {waypoints}")
    
    # 记录所有结果
    all_results = []
    
    # 为每对相邻路径点规划路径
    for i in range(len(waypoints) - 1):
        start_point = waypoints[i]
        end_point = waypoints[i+1]
        
        print(f"\n=== 规划从点{i+1}到点{i+2}的路径 ===")
        start_name = f"点{i+1}"
        end_name = f"点{i+2}"
        
        # 创建目标点云
        goal_pcd = create_goal_point_cloud(end_point)
        
        # 规划路径
        print(f"起点: {start_point}")
        print(f"终点: {end_point}")
        path = a_star_optimized(
            start_point, 
            goal_pcd, 
            obstacles, 
            grid_size=args.grid_size,
            goal_threshold=args.goal_threshold,
            collision_threshold=args.collision_threshold,
            max_time=args.max_time,
            max_steps=args.max_steps,
            z_min=args.obstacle_z_min, 
            z_max=args.obstacle_z_max
        )
        
        # 调整路径
        adjusted_path, path2d = adjust_path(path, start_point)
        
        # 保存路径
        # if args.save_json:
        #     path_file = os.path.join(args.output_dir, f'path_{i+1}_to_{i+2}.json')
        #     with open(path_file, 'w') as f:
        #         json.dump(path2d, f)
        #     print(f"路径已保存到 {path_file}")
        
        # 生成机器人指令
        # commands = generate_robot_commands(path2d, initial_direction=None, include_initial_turn=False)
        commands, initial_direction = generate_robot_commands(path2d, 
                                        initial_direction=args.initial_direction, 
                                        include_initial_turn=False)
        # 保存指令
        cmd_file = os.path.join(args.output_dir, f'commands_{i+1}_to_{i+2}.txt')
        with open(cmd_file, 'w') as f:
            for cmd in commands:
                f.write(json.dumps(cmd) + '\n')
        
        # 单独保存每条路径的可视化图（带障碍物）
        vis_file = os.path.join(args.output_dir, f'path_{i+1}_to_{i+2}.png')
        # 合并所有障碍物用于可视化
        combined_obstacle_points = []
        combined_obstacle_colors = []
        for obs_pcd in obstacles:
            points = np.asarray(obs_pcd.points)
            colors = np.asarray(obs_pcd.colors) if hasattr(obs_pcd, 'colors') and len(obs_pcd.colors) > 0 else np.ones((len(points), 3)) * 0.7
            combined_obstacle_points.append(points)
            combined_obstacle_colors.append(colors)
        
        if combined_obstacle_points:
            all_obstacle_points = np.vstack(combined_obstacle_points)
            all_obstacle_colors = np.vstack(combined_obstacle_colors)
            combined_obstacle_pcd = o3d.geometry.PointCloud()
            combined_obstacle_pcd.points = o3d.utility.Vector3dVector(all_obstacle_points)
            combined_obstacle_pcd.colors = o3d.utility.Vector3dVector(all_obstacle_colors)
        else:
            combined_obstacle_pcd = obstacle_pcd
            
        visualize_path_with_obstacles(
            path2d, 
            start_point, 
            end_point, 
            obstacle_pcd=combined_obstacle_pcd,
            save_path=vis_file,
            z_min=args.obstacle_z_min, 
            z_max=args.obstacle_z_max
        )
        
        # 打印指令
        print(f"\n从{start_name}到{end_name}的路径规划完成，路径长度: {len(path)}点")
        print("机器人指令:")
        for cmd in commands:
            print(f"  {cmd}")
        
        # 记录结果
        all_results.append({
            'segment': f'{i+1}_to_{i+2}',
            'start': start_point,
            'end': end_point,
            'path_length': len(path),
            'path2d': path2d,
            'commands': commands,
            'initial_direction': initial_direction,
        })

    # 保存所有结果
    if args.save_json:
        with open(os.path.join(args.output_dir, 'all_paths.json'), 'w') as f:
            json.dump(all_results, f)
    
    # 生成综合可视化图(所有路径+点云投影)
    combined_vis_file = os.path.join(args.output_dir, 'all_paths_with_pointcloud.png')
    # 为综合可视化合并所有障碍物
    combined_obstacle_points = []
    combined_obstacle_colors = []
    for obs_pcd in obstacles:
        points = np.asarray(obs_pcd.points)
        colors = np.asarray(obs_pcd.colors) if hasattr(obs_pcd, 'colors') and len(obs_pcd.colors) > 0 else np.ones((len(points), 3)) * 0.7
        combined_obstacle_points.append(points)
        combined_obstacle_colors.append(colors)
    
    if combined_obstacle_points:
        all_obstacle_points = np.vstack(combined_obstacle_points)
        all_obstacle_colors = np.vstack(combined_obstacle_colors)
        combined_obstacle_pcd = o3d.geometry.PointCloud()
        combined_obstacle_pcd.points = o3d.utility.Vector3dVector(all_obstacle_points)
        combined_obstacle_pcd.colors = o3d.utility.Vector3dVector(all_obstacle_colors)
    else:
        combined_obstacle_pcd = obstacle_pcd
        
    visualize_all_paths_with_pointcloud(all_results, combined_obstacle_pcd, combined_vis_file,
                                       z_min=args.obstacle_z_min, z_max=args.obstacle_z_max)
    
    # 将路径添加到点云文件中，并保存
    if args.add_to_pointcloud and all_results:
        output_ply_file = os.path.join(args.output_dir, 'cloud_map_with_paths.ply')
        add_path_to_pointcloud(
            pcd, 
            all_results, 
            z_height=0.6,
            output_file=output_ply_file,
            path_point_interval_along=0.1,
            marker_side_length=0.2,
            marker_point_spacing=0.05,
            path_line_width=0.1,
            path_line_points_across=3
        )
        print(f"已将轨迹添加到点云文件并保存为 {output_ply_file}")

    print(f"\n=== 所有路径规划完成，结果已保存到 {args.output_dir} 目录 ===")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)