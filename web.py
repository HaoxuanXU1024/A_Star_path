import json
import os
import shutil

import numpy as np
import open3d as o3d

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from astar import a_star_optimized, adjust_path, generate_robot_commands
from parsers import parse_obstacles, parse_polygon
from utils import create_goal_point_cloud, create_square_obstacle, extract_obstacles, load_point_cloud, remove_points_in_polygon
from visualization import visualize_all_paths_with_pointcloud

OUTPUT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/path_results'

def delete_contents(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 读取点云文件
    cloud_file = "/home/hpx/tyh-work/point_path/cloud_map_adjusted_m2.ply"
    pcd = load_point_cloud(cloud_file)

    default_polygon_str = "-19,-10.5;-19,-6.1;-14.2,-5.7;-14.2,-10.2"
    print(f"移除默认多边形区域: {default_polygon_str}")
    polygon = parse_polygon(default_polygon_str)
    pcd = remove_points_in_polygon(pcd, polygon)

    
    obstacle_pcd = extract_obstacles(pcd, z_min=-1.0, z_max=0.6)
    obstacles = [obstacle_pcd]

    obstacle_z_min = 0.35
    obstacle_z_max = 0.6

    additional_obstacles = parse_obstacles("-11,-7.0,0.5")
    for x, y, width in additional_obstacles:
        square_obstacle = create_square_obstacle(
            center_x=x, 
            center_y=y, 
            width=width,
            z_min=obstacle_z_min,
            z_max=obstacle_z_max,
            density=0.05
        )
        obstacles.append(square_obstacle)
        print(f"已添加方形障碍物: 中心({x:.1f}, {y:.1f})，宽度{width:.1f}m")

    app = Flask(__name__)
    CORS(app)

    @app.route('/out/<path:path>') #Everything else just goes by filename
    def sendstuff(path):
        return send_from_directory(OUTPUT_DIR, path)


    @app.route("/calculate_path", methods=["POST"])
    def calculate_path():
        delete_contents(OUTPUT_DIR)
        waypoints = request.json.get("waypoints", None)
        # check if waypoints is an array with numbers only, and length >= 2
        if not isinstance(waypoints, list) or len(waypoints) < 2:
            return jsonify({"error": "Invalid waypoints"}), 400

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
            path = a_star_optimized(start_point, goal_pcd, obstacles,
                grid_size=0.1,
                goal_threshold=0.1,
                collision_threshold=0.49,
                max_time=300,
                max_steps=300000,
                z_min=obstacle_z_min, 
                z_max=obstacle_z_max)
            
            # 调整路径
            adjusted_path, path2d = adjust_path(path, start_point)
            
            # 生成机器人指令
            commands, initial_direction = generate_robot_commands(path2d, initial_direction=270, include_initial_turn=False)
            
            # 保存指令
            cmd_file = os.path.join(OUTPUT_DIR, f'commands_{i+1}_to_{i+2}.json')
            with open(cmd_file, 'w') as f:
                json.dump(commands, f)
            
            # # 单独保存每条路径的可视化图
            # vis_file = os.path.join(output_dir, f'path_{i+1}_to_{i+2}.png')
            # visualize_path(path2d, start_point, end_point, save_path=vis_file)
            
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
        
        # 生成综合可视化图(所有路径+点云投影)
        combined_vis_file = os.path.join(OUTPUT_DIR, 'all_paths_with_pointcloud.png')

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


        mapping = visualize_all_paths_with_pointcloud(all_results, combined_obstacle_pcd, combined_vis_file,
                                        z_min=obstacle_z_min, z_max=obstacle_z_max)

        final = {
            'paths': all_results,
            'mapping': mapping,
        }

        with open(os.path.join(OUTPUT_DIR, 'all_paths.json'), 'w') as f:
            json.dump(final, f)
        
        print(f"\n=== 所有路径规划完成，结果已保存到 {OUTPUT_DIR} 目录 ===")
        return ('', 204)
    app.run(host='0.0.0.0', debug=True, use_reloader=False)

# 执行主程序
if __name__ == "__main__":
    main()