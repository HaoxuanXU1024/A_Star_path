import os
import json
import time
import argparse
from utils import (
    load_point_cloud, 
    extract_obstacles, 
    create_goal_point_cloud,
    create_pool_obstacle,
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

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='A*路径规划工具')
    
    parser.add_argument('--input', type=str, required=True,
                        help='输入的点云文件路径 (.ply)')
    
    parser.add_argument('--output_dir', type=str, default='path_results',
                        help='输出目录')
    
    parser.add_argument('--waypoints', type=str,
                        help='路径点文件 (JSON格式) 或直接的坐标列表 "x1,y1,z1;x2,y2,z2;..."')
    
    parser.add_argument('--obstacle_z_min', type=float, default=-1.0,
                        help='障碍物提取的最小Z值')
    
    parser.add_argument('--obstacle_z_max', type=float, default=0.6,
                        help='障碍物提取的最大Z值')
    
    parser.add_argument('--grid_size', type=float, default=0.1,
                        help='A*算法的网格大小')
    
    parser.add_argument('--goal_threshold', type=float, default=0.1,
                        help='目标判定阈值')
    
    parser.add_argument('--collision_threshold', type=float, default=0.05,
                        help='碰撞判定阈值')
    
    parser.add_argument('--turn_penalty', type=float, default=20,
                        help='转向惩罚系数')
    
    parser.add_argument('--remove_polygon', action='store_true',
                        help='是否移除默认的多边形区域 (默认区域: -19,-10.5;-19,-6.1;-14.2,-5.7;-14.2,-10.2)')

    parser.add_argument('--initial_direction', type=int, default=None, 
                        help='机器人的初始朝向角度(0-359)，为None则使用第一段路径的方向作为初始方向')
       
    parser.add_argument('--save_json', action='store_true',
                        help='是否保存路径的JSON文件')

    parser.add_argument('--add_to_pointcloud', action='store_true',
                        help='是否将路径添加到点云文件')
    
    parser.add_argument('--max_time', type=int, default=300,
                        help='A*算法最大搜索时间（秒）')
    
    parser.add_argument('--max_steps', type=int, default=300000,
                        help='A*算法最大搜索步数')
    
    return parser.parse_args()

def parse_waypoints(waypoints_str):
    """
    解析路径点字符串或JSON文件
    
    Args:
        waypoints_str: 路径点字符串 (x1,y1,z1;x2,y2,z2;...) 或 JSON文件路径
        
    Returns:
        waypoints: 路径点列表 [(x1,y1,z1), (x2,y2,z2), ...]
    """
    # 检查是否为JSON文件
    if waypoints_str.endswith('.json'):
        with open(waypoints_str, 'r') as f:
            waypoints = json.load(f)
        return waypoints
    
    # 否则认为是直接的坐标列表
    waypoints = []
    for point_str in waypoints_str.split(';'):
        coords = [float(x) for x in point_str.split(',')]
        if len(coords) != 3:
            raise ValueError(f"路径点格式错误，应为 'x,y,z': {point_str}")
        waypoints.append(tuple(coords))
    
    return waypoints

def parse_polygon(polygon_str):
    """
    解析多边形字符串
    
    Args:
        polygon_str: 多边形字符串 (x1,y1;x2,y2;x3,y3;...)
        
    Returns:
        polygon: 多边形顶点列表 [(x1,y1), (x2,y2), ...]
    """
    polygon = []
    for point_str in polygon_str.split(';'):
        coords = [float(x) for x in point_str.split(',')]
        if len(coords) != 2:
            raise ValueError(f"多边形顶点格式错误，应为 'x,y': {point_str}")
        polygon.append(tuple(coords))
    
    return polygon

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
    
    # 解析路径点
    if args.waypoints:
        waypoints = parse_waypoints(args.waypoints)
    else:
        # 默认路径点示例
        waypoints = [
            (-9.8, -4.0, 1.0),
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
            max_steps=args.max_steps
        )
        
        # 调整路径
        adjusted_path, path2d = adjust_path(path, start_point)
        
        # 保存路径
        if args.save_json:
            path_file = os.path.join(args.output_dir, f'path_{i+1}_to_{i+2}.json')
            with open(path_file, 'w') as f:
                json.dump(path2d, f)
            print(f"路径已保存到 {path_file}")
        
        # 生成机器人指令
        # commands = generate_robot_commands(path2d, initial_direction=None, include_initial_turn=False)
        commands = generate_robot_commands(path2d, 
                                        initial_direction=args.initial_direction, 
                                        include_initial_turn=False)
        # 保存指令
        cmd_file = os.path.join(args.output_dir, f'commands_{i+1}_to_{i+2}.txt')
        with open(cmd_file, 'w') as f:
            for cmd in commands:
                f.write(cmd + '\n')
        
        # 单独保存每条路径的可视化图（带障碍物）
        vis_file = os.path.join(args.output_dir, f'path_{i+1}_to_{i+2}.png')
        visualize_path_with_obstacles(
            path2d, 
            start_point, 
            end_point, 
            obstacle_pcd=obstacle_pcd,
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
            'commands': commands
        })
    
    # 保存所有结果
    if args.save_json:
        with open(os.path.join(args.output_dir, 'all_paths.json'), 'w') as f:
            json.dump(all_results, f)
    
    # 生成综合可视化图(所有路径+点云投影)
    combined_vis_file = os.path.join(args.output_dir, 'all_paths_with_pointcloud.png')
    visualize_all_paths_with_pointcloud(all_results, obstacle_pcd, combined_vis_file)
    
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