"""路径可视化脚本

- 显示模型生成的智能体轨迹（红色实线）
- 在凹形(C)和狭窄(D)地图中启用固定动态障碍物（由 map_generator.py 提供）
- 记录并绘制动态障碍物轨迹
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from env import AutonomousNavEnv
from model import CascadedDualAttentionActorCritic
from global_planner import SmartAStarPlanner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


class PathVisualizer:
    def __init__(self, model_path='best_navigation_model.pth'):
        self.model = CascadedDualAttentionActorCritic(action_dim=2).to(device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"已加载模型: {model_path}")
        else:
            print(f"警告: 模型文件不存在 {model_path}，使用随机初始化")
        
        self.model.eval()
    
    def select_action(self, obs_dict, hidden_state):
        with torch.no_grad():
            img = torch.FloatTensor(obs_dict['image']).unsqueeze(0).to(device)
            vec = torch.FloatTensor(obs_dict['vector']).unsqueeze(0).to(device)
            mu, std, val, next_hidden = self.model({'image': img, 'vector': vec}, hidden_state)
            action = mu  # 使用确定性策略
        return action.cpu().numpy().flatten(), next_hidden
    
    def generate_path_with_dynamic_obs(self, env, max_steps=500):
        """
        生成路径，同时记录动态障碍物位置
        """
        obs_dict, _ = env.reset()
        
        hidden_state = torch.zeros(1, 256).to(device)
        
        trajectory = [env.agent_pos.copy()]
        dynamic_obs_history = []  # 记录动态障碍物的轨迹
        
        # 记录初始动态障碍物位置
        if env.dynamic_obstacles:
            dynamic_obs_history.append([
                {'pos': obs['pos'].copy(), 'radius': float(obs.get('radius', 2.0))}
                for obs in env.dynamic_obstacles
            ])
        
        success = False
        for step in range(max_steps):
            action, next_hidden = self.select_action(obs_dict, hidden_state)
            obs_dict, reward, done, _, info = env.step(action)
            hidden_state = next_hidden
            
            trajectory.append(env.agent_pos.copy())
            
            # 记录动态障碍物位置
            if env.dynamic_obstacles:
                dynamic_obs_history.append([
                    {'pos': obs['pos'].copy(), 'radius': float(obs.get('radius', 2.0))}
                    for obs in env.dynamic_obstacles
                ])
            
            if done:
                success = info.get('success', False)
                break
        
        return np.array(trajectory), dynamic_obs_history, success
    
    def visualize_all_maps(self, save_path='optimal_paths.png'):
        """在4种地图上可视化模型生成的路径"""
        print("\n生成路径可视化...")
        
        map_types = ['simple', 'complex', 'concave', 'narrow']
        map_names = ['(a) 简单环境', '(b) 复杂环境', '(c) 凹形障碍', '(d) 狭窄通道']
        
        # concave / narrow 的动态障碍物由 env <- map_generator 固定提供
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, (map_type, map_name) in enumerate(zip(map_types, map_names)):
            ax = axes[i]
            
            # 创建环境
            env = AutonomousNavEnv(map_type=map_type)
            planner = SmartAStarPlanner(env.static_map)

            # 先规划一次全局路径并注入到观测（第2通道），让模型用得到
            try:
                gp = planner.plan(env.start_pos, env.goal_pos, env.dynamic_obstacles)
                if gp is not None and len(gp) > 0:
                    env.set_global_path(gp)
            except:
                pass

            # 生成路径
            trajectory, dyn_obs_history, success = self.generate_path_with_dynamic_obs(env, max_steps=500)
            
            # 计算A*全局路径（用于展示）
            try:
                global_path = planner.plan(env.start_pos, env.goal_pos, env.dynamic_obstacles)
            except:
                global_path = None
            
            # 绘制地图（黑色障碍物）
            ax.imshow(env.static_map.T, cmap='Greys', origin='lower', extent=[0, 80, 0, 80])
            
            # 绘制A*全局路径（蓝色虚线）
            if global_path is not None and len(global_path) > 0:
                global_path = np.array(global_path)
                ax.plot(global_path[:, 0], global_path[:, 1], 'b--', 
                       linewidth=1.5, alpha=0.6, label='A* 全局路径')
            
            # 绘制动态障碍物（如果有）
            if dyn_obs_history and len(dyn_obs_history) > 0:
                # 绘制动态障碍物的轨迹（浅蓝色虚线）
                for obs_idx in range(len(dyn_obs_history[0])):
                    obs_traj = np.array([frame[obs_idx]['pos'] for frame in dyn_obs_history])
                    radius0 = float(dyn_obs_history[0][obs_idx].get('radius', 2.0))
                    ax.plot(obs_traj[:, 0], obs_traj[:, 1], 'c--', 
                           linewidth=1.5, alpha=0.5)
                    
                    # 绘制动态障碍物的初始位置（蓝色圆圈）
                    circle_start = plt.Circle(
                        (obs_traj[0][0], obs_traj[0][1]),
                        radius0, color='blue', alpha=0.7, label='动态障碍物' if obs_idx == 0 else None
                    )
                    ax.add_patch(circle_start)
                    
                    # 绘制动态障碍物的最终位置（浅蓝色圆圈）
                    circle_end = plt.Circle(
                        (obs_traj[-1][0], obs_traj[-1][1]),
                        radius0, color='cyan', alpha=0.5
                    )
                    ax.add_patch(circle_end)
            
            # 绘制智能体轨迹（红色实线）
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', 
                   linewidth=2.5, label='智能体轨迹')
            
            # 绘制起点（绿色圆点）
            ax.scatter(env.start_pos[0], env.start_pos[1], 
                      c='green', s=200, marker='o', zorder=5, 
                      edgecolors='black', linewidth=2, label='起点')
            
            # 绘制终点（红色五角星）
            ax.scatter(env.goal_pos[0], env.goal_pos[1], 
                      c='red', s=300, marker='*', zorder=5, 
                      edgecolors='black', linewidth=1, label='终点')
            
            # 设置标题
            status = "成功到达" if success else "失败/碰撞"
            ax.set_title(f'{map_name}\n{status} | 步数: {len(trajectory)}', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 80)
            ax.set_ylim(0, 80)
            ax.set_aspect('equal')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"路径可视化已保存: {save_path}")


def main():
    print("=" * 60)
    print("模型路径可视化")
    print("=" * 60)
    
    # 创建可视化器
    visualizer = PathVisualizer('best_navigation_model.pth')
    
    # 生成可视化
    visualizer.visualize_all_maps('optimal_paths.png')
    
    print("\n完成！")
    
    # 打开图片
    os.system('start optimal_paths.png')


if __name__ == '__main__':
    main()
