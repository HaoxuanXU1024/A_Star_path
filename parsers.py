import argparse

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='A*路径规划工具')

    parser.add_argument('--input', type=str, required=True,
                        help='输入的点云文件路径 (.ply)')

    parser.add_argument('--output_dir', type=str, default='path_results',
                        help='输出目录')

    parser.add_argument('--waypoints', type=str,
                        help='路径点文件 (JSON格式) 或直接的坐标列表 "x1,y1,z1;x2,y2,z2;..."')

    parser.add_argument('--obstacle_z_min', type=float, default=0.35,
                        help='障碍物提取的最小Z值')

    parser.add_argument('--obstacle_z_max', type=float, default=0.6,
                        help='障碍物提取的最大Z值')

    parser.add_argument('--grid_size', type=float, default=0.1,
                        help='A*算法的网格大小')

    parser.add_argument('--goal_threshold', type=float, default=0.1,
                        help='目标判定阈值')

    parser.add_argument('--collision_threshold', type=float, default=0.05,
                        help='碰撞判定阈值')

    parser.add_argument('--turn_penalty', type=float, default=10,
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

    parser.add_argument('--add_obstacles', type=str,
                        help='添加额外障碍物 格式: "x1,y1,width1;x2,y2,width2;..." 例如: "-12.8,-11.0,1.0"')

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

def parse_obstacles(obstacles_str):
    """
    解析额外障碍物字符串
    
    Args:
        obstacles_str: 障碍物字符串 (x1,y1,width1;x2,y2,width2;...)
        
    Returns:
        obstacles: 障碍物列表 [(x1,y1,width1), (x2,y2,width2), ...]
    """
    obstacles = []
    for obstacle_str in obstacles_str.split(';'):
        coords = [float(x) for x in obstacle_str.split(',')]
        if len(coords) != 3:
            raise ValueError(f"障碍物格式错误，应为 'x,y,width': {obstacle_str}")
        obstacles.append(tuple(coords))

    return obstacles
