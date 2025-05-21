# A* 路径规划工具

基于点云的A*路径规划工具，可用于复杂环境中的机器人导航路径生成。

## 功能特点

- 基于点云数据的障碍物识别
- 优化的A*算法，支持自适应网格大小
- 考虑转向惩罚的路径平滑
- 可视化路径和障碍物
- 生成机器人运动指令
- 支持将路径添加到点云文件中

## 目录结构

```
astar_path/
├── astar.py       # A*算法及相关函数
├── utils.py       # 点云处理和辅助功能
├── visualization.py # 可视化功能
├── main.py        # 主程序入口
└── README.md      # 项目说明
```

## 安装依赖

本项目依赖以下Python库：

```bash
pip install numpy open3d matplotlib scipy
```

## 使用方法

### 基本用法

```bash
python main.py --input <点云文件路径> --output_dir <输出目录> --waypoints <路径点>
```

### 示例

完整效果：

```bash
python main.py --input G:/HKUST/cloud_map_adjusted_m2.ply  --grid_size 0.1 --collision_threshold 0.05 --turn_penalty 20 --remove_polygon --initial_direction 180
```

指定路径点（格式为"x1,y1,z1;x2,y2,z2;..."）：

```bash
python main.py --input cloud_map.ply --waypoints "-9.8,-4.0,1.0;-15.5,-8.0,1.0"
```

移除点云中的特定区域：

```bash
python main.py --input cloud_map.ply --remove_polygon
```

保存JSON文件：

```bash
python main.py --input cloud_map.ply --save_json
```
### 主要参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--input` | 必填 | 输入点云文件路径(.ply) |
| `--output_dir` | path_results | 输出目录 |
| `--waypoints` | 默认路径点 | 路径点列表或JSON文件 |
| `--obstacle_z_min` | -1.0 | 障碍物提取的最小Z值 |
| `--obstacle_z_max` | 0.6 | 障碍物提取的最大Z值 |
| `--grid_size` | 0.1 | A*算法的网格大小 |
| `--goal_threshold` | 0.1 | 目标判定阈值 |
| `--collision_threshold` | 0.05 | 碰撞判定阈值 |
| `--turn_penalty` | 0.2 | 转向惩罚系数 |
| `--remove_polygon` | None | 要移除的多边形区域 |
| `--add_to_pointcloud` | False | 是否将路径添加到点云文件 |
| `--max_time` | 300 | A*算法最大搜索时间（秒） |
| `--max_steps` | 300000 | A*算法最大搜索步数 |

## 输出结果

程序会在指定的输出目录中生成以下文件：

- `path_1_to_2.json`：规划的路径点（相对坐标）
- `commands_1_to_2.txt`：机器人运动指令
- `path_1_to_2.png`：单条路径可视化图
- `all_paths.json`：所有路径信息
- `all_paths_with_pointcloud.png`：所有路径与点云的综合可视化图
- `cloud_map_with_paths.ply`：包含路径的点云文件（如果指定了`--add_to_pointcloud`）

## 自定义

- 可以在`astar.py`中调整A*算法的参数和行为
- 可以在`utils.py`中添加更多点云处理功能
- 可以在`visualization.py`中自定义可视化样式