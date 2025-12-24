import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

# 导入你的模块
from map_generator import MapGenerator
from global_planner import SmartAStarPlanner

def generate_dynamic_obstacles(grid_map, num=8):
    """ 模拟 env.py 中的动态障碍物生成逻辑 """
    rows, cols = grid_map.shape
    obstacles = []
    attempts = 0
    while len(obstacles) < num and attempts < 1000:
        # 在地图中间区域随机生成
        pos = np.random.uniform(10, rows-10, 2)
        # 检查是否在静态障碍物上 (保持一定距离)
        xi, yi = int(pos[0]), int(pos[1])
        if grid_map[xi, yi] == 0:
            # 简单的速度向量
            vel = np.random.uniform(-0.5, 0.5, 2)
            obstacles.append({'pos': pos, 'vel': vel})
        attempts += 1
    return obstacles

def plot_scenario(ax, map_type, title_label):
    # 1. 获取地图
    map_gen = MapGenerator(map_size=80)
    grid, start, goal = map_gen.get_map(map_type)
    
    # 2. 生成动态障碍物
    # 针对不同地图调整障碍物数量，为了展示效果
    num_obs = 10 if map_type in ['complex', 'concave'] else 6
    dyn_obs = generate_dynamic_obstacles(grid, num=num_obs)
    
    # 3. A* 规划 (考虑动态障碍物膨胀代价)
    planner = SmartAStarPlanner(grid)
    path = planner.plan(start, goal, dyn_obs)
    
    # --- 开始绘图 ---
    
    # A. 绘制静态地图 (黑色=墙，白色=路)
    # 使用自定义 colormap: 0=white, 1=black
    cmap = ListedColormap(['white', 'black'])
    ax.imshow(grid.T, cmap=cmap, origin='lower', extent=[0, 80, 0, 80])
    
    # B. 绘制动态障碍物 (蓝色圆点 + 速度箭头)
    for obs in dyn_obs:
        # 画圆点
        circle = patches.Circle(obs['pos'], radius=1.5, edgecolor='blue', facecolor='cyan', alpha=0.8, zorder=5, label='Dynamic Obs')
        ax.add_patch(circle)
        # 画速度箭头 (表示移动意图)
        ax.arrow(obs['pos'][0], obs['pos'][1], obs['vel'][0]*3, obs['vel'][1]*3, 
                 head_width=1.5, head_length=1.5, fc='blue', ec='blue', zorder=6)
        # 画一个淡淡的"危险圈" (对应A*的膨胀代价)
        danger_zone = patches.Circle(obs['pos'], radius=4.0, edgecolor='none', facecolor='red', alpha=0.15, zorder=4)
        ax.add_patch(danger_zone)

    # C. 绘制全局路径 (红色实线)
    if len(path) > 0:
        ax.plot(path[:, 0], path[:, 1], color='red', linewidth=2.5, linestyle='-', label='Global Path (A*)', zorder=10)
        # 画路径点 (可选)
        # ax.scatter(path[:, 0], path[:, 1], c='orange', s=10, zorder=11)

    # D. 绘制起点和终点
    ax.scatter(start[0], start[1], c='green', marker='o', s=150, edgecolors='black', label='Start', zorder=20)
    ax.scatter(goal[0], goal[1], c='purple', marker='*', s=250, edgecolors='black', label='Goal', zorder=20)

    # 美化设置
    ax.set_title(title_label, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)
    ax.set_xticks([]) # 去掉坐标刻度，更像示意图
    ax.set_yticks([])
    
    # 只在第一个图显示图例，防止遮挡
    if map_type == 'simple':
        # 创建自定义图例句柄
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2.5, label='A* Path'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='purple', markersize=15, label='Goal'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markeredgecolor='blue', markersize=10, label='Dynamic Obs'),
            patches.Patch(facecolor='black', label='Static Obs')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

def main():
    # 创建画布
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    scenarios = [
        ('simple', '(a) Simple Environment'),
        ('complex', '(b) Complex Environment'),
        ('concave', '(c) Concave (U-Trap) Environment'),
        ('narrow', '(d) Narrow Corridor Environment')
    ]
    
    print("正在生成论文可视化图表...")
    
    for i, (map_type, label) in enumerate(scenarios):
        print(f"  - Rendering {map_type}...")
        plot_scenario(axes[i], map_type, label)
    
    plt.tight_layout()
    
    # 保存为高清图片 (300 DPI)
    save_path = 'paper_visualization_full.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图片已保存至: {save_path}")
    # plt.show() # 如果你在本地运行，可以取消注释这一行来查看窗口

if __name__ == "__main__":
    main()