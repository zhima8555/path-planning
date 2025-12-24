"""
可视化地图和目标位置
"""
import numpy as np
import matplotlib.pyplot as plt
from env import PortGridEnv
from global_planner import AStarPlanner

def visualize_map_and_goals():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    difficulties = [0.0, 0.5, 1.0]
    goal_names = ["Easy (25,25)", "Medium (45,45)", "Hard (70,70)"]
    
    for idx, (diff, goal_name) in enumerate(zip(difficulties, goal_names)):
        env = PortGridEnv(difficulty=diff)
        planner = AStarPlanner(env.static_map)
        
        env.reset()
        path = planner.plan(env.agent_pos, env.goal_pos)
        
        ax = axes[idx]
        
        # 显示静态地图
        ax.imshow(env.static_map.T, cmap='gray_r', origin='lower', alpha=0.5)
        
        # 标记起点和终点
        start_x, start_y = env.agent_pos
        goal_x, goal_y = env.goal_pos
        
        ax.plot(start_x, start_y, 'go', markersize=15, label='Start', markeredgecolor='black', markeredgewidth=2)
        ax.plot(goal_x, goal_y, 'r*', markersize=20, label='Goal', markeredgecolor='black', markeredgewidth=2)
        
        # 检查目标位置是否在障碍物内
        goal_is_obstacle = env.static_map[int(goal_x), int(goal_y)] == 1.0
        
        # 显示路径
        if len(path) > 0:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, alpha=0.7, label=f'Path (len={len(path)})')
        else:
            ax.text(40, 5, 'NO PATH FOUND!', fontsize=14, color='red', 
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        ax.set_title(f'{goal_name}\nDifficulty={diff:.1f}, Path len={len(path)}\n'
                    f'Goal in obstacle: {goal_is_obstacle}', fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, env.map_size)
        ax.set_ylim(0, env.map_size)
    
    plt.tight_layout()
    plt.savefig('map_visualization.png', dpi=200)
    print("Saved: map_visualization.png")
    plt.close()
    
    # 详细检查目标位置
    print("\n=== 详细检查目标位置 ===")
    for diff, goal_name in zip(difficulties, goal_names):
        env = PortGridEnv(difficulty=diff)
        goal_x, goal_y = int(env.goal_pos[0]), int(env.goal_pos[1])
        is_blocked = env.static_map[goal_x, goal_y] == 1.0
        print(f"{goal_name}: pos=({goal_x},{goal_y}), blocked={is_blocked}")
        
        # 检查目标周围3x3区域
        blocked_count = 0
        for i in range(max(0, goal_x-1), min(env.map_size, goal_x+2)):
            for j in range(max(0, goal_y-1), min(env.map_size, goal_y+2)):
                if env.static_map[i, j] == 1.0:
                    blocked_count += 1
        print(f"  Blocked cells in 3x3 area: {blocked_count}/9")

if __name__ == "__main__":
    visualize_map_and_goals()
