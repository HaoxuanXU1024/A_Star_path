import heapq
import math
import numpy as np
import time
import random
from scipy.spatial import KDTree

def a_star_optimized(start, goal_pcd, obstacles, grid_size=0.1, goal_threshold=0.1, 
                     collision_threshold=0.05, max_time=300, max_steps=300000,
                     z_min=-2.0, z_max=0.5):
    """
    优化的A*路径规划算法，适用于复杂和远距离场景。
    
    Args:
        start: 起始点坐标 (x, y, z)
        goal_pcd: 目标点云
        obstacles: 障碍物点云列表
        grid_size: 网格大小
        goal_threshold: 目标判定阈值
        collision_threshold: 碰撞判定阈值
        max_time: 最大搜索时间（秒）
        max_steps: 最大搜索步数
        z_min: 障碍物高度筛选下限
        z_max: 障碍物高度筛选上限
    
    Returns:
        规划路径点列表
    """
    # 1. 预处理阶段 - 构建障碍物KD树并优化
    print("构建障碍物KD树...")
    start_time = time.time()
    obstacle_points = []
    for pcd in obstacles:
        points = np.asarray(pcd.points)
        # 只保留与起点高度相近的点
        mask = (points[:, 2] >= (start[2] + z_min)) & (points[:, 2] <= (start[2] + z_max))
        filtered_points = points[mask, :2]  # 只保留x,y坐标
        if len(filtered_points) > 0:  # 避免添加空的点集
            obstacle_points.append(filtered_points)
    
    # 合并所有障碍物点并构建KD树
    if obstacle_points and len(obstacle_points) > 0:
        all_obstacle_points = np.vstack(obstacle_points)
        # 使用voxel降采样减少点数量，提高KD树效率
        voxel_size = 0.15
        voxel_grid = {}
        for point in all_obstacle_points:
            voxel_idx = tuple(np.floor(point / voxel_size).astype(int))
            voxel_grid[voxel_idx] = point
        all_obstacle_points = np.array(list(voxel_grid.values()))
        
        obstacle_kdtree = KDTree(all_obstacle_points)
        print(f"KD树构建完成，包含 {len(all_obstacle_points)} 个点，耗时 {time.time() - start_time:.2f}s")
    else:
        print("警告：没有找到障碍物点！")
        obstacle_kdtree = None
    
    # 2. 目标处理 - 计算目标中心和搜索边界
    goal_points = np.asarray(goal_pcd.points)
    goal_center = np.mean(goal_points, axis=0)[:2]  # 只取x,y
    
    # 计算搜索边界，限制搜索空间
    min_x = min(start[0], goal_center[0]) - 5.0
    max_x = max(start[0], goal_center[0]) + 5.0
    min_y = min(start[1], goal_center[1]) - 5.0
    max_y = max(start[1], goal_center[1]) + 5.0
    search_bounds = (min_x, max_x, min_y, max_y)
    
    # 3. 优化的辅助函数
    def is_within_bounds(point, bounds):
        """检查点是否在搜索边界内"""
        x, y = point[:2]
        min_x, max_x, min_y, max_y = bounds
        return min_x <= x <= max_x and min_y <= y <= max_y
    
    def adaptive_grid_size(point, base_size=grid_size):
        """更好的自适应网格大小函数，对点云缺失区域使用较大步长"""
        if obstacle_kdtree is None:
            return base_size * 2.0  # 增大系数，无障碍物时使用更大步长
        
        try:
            distance, _ = obstacle_kdtree.query([point[0], point[1]], k=1)
            # 更激进的步长策略：远离障碍物时大步走
            return max(0.2, min(1.0, base_size * (1.0 + distance)))
        except:
            # 查询失败时使用较大步长
            return base_size * 2.0
    
    def get_sorted_directions(current, goal):
        """根据朝向目标的角度排序方向"""
        dx = goal[0] - current[0]
        dy = goal[1] - current[1]
        target_angle = math.degrees(math.atan2(dy, dx)) % 360
        
        # 计算每个方向与目标方向的角度差
        dir_diffs = []
        for d in directions:
            angle_diff = abs((d - target_angle) % 360)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            dir_diffs.append((d, angle_diff))
        
        # 按角度差排序，优先尝试朝向目标的方向
        return [d for d, _ in sorted(dir_diffs, key=lambda x: x[1])]

    def heuristic(a, b, weight=1.0):
        """加权启发式函数，增强对目标的导向性"""
        # 计算曼哈顿距离
        manhattan_dist = (
            abs(a[0] - b[0]) +  # x轴距离
            abs(a[1] - b[1])    # y轴距离
        )
        return weight * manhattan_dist
    
    def is_clear_path(start, end, check_steps=10):
        """检查两点之间是否有无障碍直线路径"""
        if obstacle_kdtree is None:
            return True
            
        vec = np.array(end) - np.array(start)
        distance = np.linalg.norm(vec)
        if distance < 0.01:  # 距离太小
            return True
            
        vec = vec / distance  # 单位向量
        
        # 沿直线检查几个点
        for i in range(1, check_steps):
            t = i / check_steps
            check_point = start + t * vec
            
            try:
                dist, _ = obstacle_kdtree.query(check_point[:2], k=1)
                if dist < collision_threshold:
                    return False  # 有障碍
            except:
                continue
                
        return True  # 全部通过，无障碍

    def is_collision(point, threshold=collision_threshold):
        """碰撞检测函数，使用缓存和早期返回策略"""
        if obstacle_kdtree is None:
            return False
        x, y, _ = point
        # 查询KD树中距离最近的点
        distance, _ = obstacle_kdtree.query([x, y], k=1)
        return distance < threshold
    
    def is_goal(point, threshold=goal_threshold):
        """判断是否达到目标"""
        x, y, _ = point
        distances = np.linalg.norm(goal_points[:, :2] - [x, y], axis=1)
        return np.any(distances < threshold)
    
    # 4. A*算法主体逻辑
    print(f"开始路径规划，起点: {start[:2]}, 目标中心: {goal_center}")
    
    # 更多方向选择，使路径更平滑
    directions = [i * 45 for i in range(8)]  # 8方向，每45°一个方向
    
    # A*相关数据结构初始化
    start_state = (start[0], start[1], 0)  # 初始状态，方向设为0°
    opened = []
    closed = set()  # 记录已访问状态，加速搜索
    heapq.heappush(opened, (0, start_state))
    came_from = {}
    g_score = {start_state: 0}
    f_score = {start_state: heuristic(start_state[:2], goal_center)}
    closest_state = start_state
    closest_dist = heuristic(start_state[:2], goal_center)
    best_path_cost = float('inf')
    
    step_count = 0
    search_start_time = time.time()
    
    # 主循环
    while opened and step_count < max_steps:
        # 检查是否超时
        if time.time() - search_start_time > max_time:
            print(f"搜索超时，已用时: {time.time() - search_start_time:.2f}s")
            break
        
        step_count += 1
        if step_count % 1000 == 0:
            print(f"搜索步数: {step_count}, 开放列表长度: {len(opened)}, 耗时: {time.time() - search_start_time:.2f}s")
        
        # 取出评分最低的状态
        current_f, current = heapq.heappop(opened)
        
        # 如果已处理过该状态，跳过
        if current in closed:
            continue
        closed.add(current)
        
        # 检查是否到达目标
        if is_goal(current):
            # 回溯并构建路径
            path = []
            while current in came_from:
                path.append((current[0], current[1], start[2]))
                current = came_from[current]
            path.append((start[0], start[1], start[2]))
            path.reverse()
            print(f"成功找到目标! 路径长度: {len(path)}, 总步数: {step_count}, 总耗时: {time.time() - search_start_time:.2f}s")
            return path
        
        # 获取当前位置的自适应网格大小
        current_grid_size = adaptive_grid_size(current)
        
        # 获取按照目标方向排序的方向列表
        sorted_directions = get_sorted_directions(current, goal_center)
        
        # 扩展邻居状态
        for d in sorted_directions:
            rad = math.radians(d)
            nx = current[0] + current_grid_size * math.cos(rad)
            ny = current[1] + current_grid_size * math.sin(rad)
            neighbor = (nx, ny, d)
            
            # 边界检查
            if not is_within_bounds((nx, ny), search_bounds):
                continue
            
            # 如果邻居已访问过，跳过
            if neighbor in closed:
                continue
            
            # 碰撞检测
            if is_collision(neighbor):
                continue
            
            # 计算新的代价评分
            tentative_g = g_score[current] + cost_with_turn_penalty(current, neighbor, turn_penalty=10)
            
            # 剪枝：如果到达当前节点的成本已经超过了到达最佳路径的成本，跳过
            if tentative_g > best_path_cost:
                continue
                
            # 如果找到更好的路径，更新
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                
                # 计算启发式评分
                h = heuristic(neighbor[:2], goal_center)
                f_score[neighbor] = tentative_g + h
                
                # 添加到开放列表
                heapq.heappush(opened, (f_score[neighbor], neighbor))
                
                # 更新最接近目标的状态
                if h < closest_dist:
                    closest_state = neighbor
                    closest_dist = h
                    
                    # 如果非常接近目标，调整目标阈值
                    if closest_dist < goal_threshold * 2:
                        best_path_cost = g_score[neighbor] + closest_dist
    
    # 搜索失败，返回到最接近目标的路径
    print(f"无法找到完整路径，尝试构建部分路径...")
    path = []
    curr = closest_state
    while curr in came_from:
        path.append((curr[0], curr[1], start[2]))
        curr = came_from[curr]
    path.append((start[0], start[1], start[2]))
    path.reverse()
    print(f"返回最佳部分路径，长度: {len(path)}, 与目标最近距离: {closest_dist:.2f}")
    return path

def cost_with_turn_penalty(current, neighbor, turn_penalty=0.2):
    """
    计算移动代价，包含转向惩罚
    
    Args:
        current: 当前状态 (x, y, direction)
        neighbor: 邻居状态 (x, y, direction)
        turn_penalty: 转向惩罚系数，越大转向代价越高
        
    Returns:
        总移动代价
    """
    # 计算距离代价
    pos_dist = np.linalg.norm(np.array(current[:2]) - np.array(neighbor[:2]))
    
    # 计算方向差（角度）
    angle_diff = abs((neighbor[2] - current[2]) % 360)
    if angle_diff > 180:  # 取最小角度差
        angle_diff = 360 - angle_diff
    
    # 基于角度差的转向惩罚
    dir_penalty = (angle_diff / 90.0) * turn_penalty
    
    return pos_dist + dir_penalty

def adjust_path(path, initial_position):
    """
    调整路径坐标，转换为相对于初始位置的坐标
    
    Args:
        path: 原始路径点列表
        initial_position: 初始位置坐标
        
    Returns:
        adjusted_path: 调整后的3D路径
        path2d: 调整后的2D路径
    """
    adjusted_path = []
    path2d = []
    for coord in path:
        adjusted_coord = (
            coord[0] - initial_position[0],
            coord[1] - initial_position[1],
            coord[2]  # 保持z坐标不变
        )
        adjusted_path.append(adjusted_coord)
        path2d.append((adjusted_coord[0], adjusted_coord[1]))
    return adjusted_path, path2d

def generate_robot_commands(path2d, initial_direction=180, include_initial_turn=False):
    """
    将二维路径点转换为机器人移动和转向指令，使用八方向（每45度一个方向）。
    
    Args:
        path2d: 二维路径点列表 [(x1,y1), (x2,y2), ...]
        initial_direction: 初始朝向角度（度），如果为None则使用第一段路径的方向作为初始方向
        include_initial_turn: 是否包含初始转向指令
        
    Returns:
        commands: 机器人指令列表
    """
    import math

    if not path2d or len(path2d) < 2:
        return []

    commands = []
    move_distance = 0.0  # 累积的直线移动距离
    EPSILON = 1e-6  # 浮点数比较的容差值

    # 获取方向的函数，直接返回角度
    def get_movement_angle(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.degrees(math.atan2(dy, dx)) % 360
        
        # 将角度量化到最近的45度方向
        return 45 * round(angle / 45) % 360

    def calculate_turn(current_dir, desired_dir):
        difference = (desired_dir - current_dir) % 360
        return {
            "type": "turn",
            "value": difference
        }

    # 确定初始方向
    if len(path2d) >= 2:
        first_point = path2d[0]
        second_point = path2d[1]
        first_segment_angle = get_movement_angle(first_point, second_point)
        
        # 如果initial_direction为None，使用第一段路径的方向作为初始方向
        current_direction = first_segment_angle if initial_direction is None else initial_direction
        
        # 是否需要包含初始转向
        if include_initial_turn and initial_direction is not None:
            turn = calculate_turn(initial_direction, first_segment_angle)
            if turn:
                commands.append(turn)
                current_direction = first_segment_angle
    else:
        # 如果路径少于2个点，使用提供的初始方向或默认为0
        current_direction = initial_direction if initial_direction is not None else 0
    
    # 跟踪上一个移动方向的角度
    prev_angle = current_direction
    
    # 主处理循环：处理每个路径段
    for i in range(1, len(path2d)):
        prev_point = path2d[i-1]
        curr_point = path2d[i]
        
        # 计算此段路径的方向角度
        curr_angle = get_movement_angle(prev_point, curr_point)
        
        # 检查是否需要转向
        angle_change = abs((curr_angle - prev_angle) % 360)
        if angle_change > 180:  # 确保使用最小角度
            angle_change = 360 - angle_change
        
        # 如果方向变化明显，需要先转向
        if angle_change >= 20:  # 使用20度作为阈值判断方向是否改变
            # 先完成当前的直行
            if move_distance > EPSILON:
                commands.append({
                    "type": "forward",
                    "value": move_distance
                })
                move_distance = 0.0
            
            # 计算并执行转向
            turn = calculate_turn(prev_angle, curr_angle)
            if turn:
                commands.append(turn)
                prev_angle = curr_angle  # 更新当前朝向
        
        # 计算并累加这段路径的距离
        distance = math.hypot(curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
        move_distance += distance
    
    # 处理最后一段直行
    if move_distance > EPSILON:
        commands.append({
            "type": "forward",
            "value": move_distance
        })    

    return commands