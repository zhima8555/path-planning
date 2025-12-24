import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from train import main as train_single_seed

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 多种子训练
def train_multiple_seeds(seeds=[42, 123, 456, 789, 999]):
    """
    使用多个随机种子训练，取平均结果
    """
    print("=" * 60)
    print("多随机种子PPO训练")
    print(f"随机种子: {seeds}")
    print("=" * 60)

    all_rewards = []
    all_losses = []

    # 保存原始配置
    original_model_dir = './models'

    for i, seed in enumerate(seeds):
        print(f"\n>>> 训练种子 {seed} ({i+1}/{len(seeds)})")
        print("-" * 50)

        # 设置随机种子
        set_seed(seed)

        # 为每个种子创建独立的模型目录
        os.makedirs(f'./models_seed_{seed}', exist_ok=True)

        # 修改全局的模型保存路径
        # 这里需要修改train.py中的保存逻辑

        # 训练（需要修改train.py以返回历史数据）
        # eval_history, loss_history = train_single_seed(seed=seed)

        # all_rewards.append(eval_history)
        # all_losses.append(loss_history)

    # 计算平均值和标准差
    if all_rewards:
        avg_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)
        avg_losses = np.mean(all_losses, axis=0)
        std_losses = np.std(all_losses, axis=0)

        # 绘制平均曲线
        plot_multi_seed_curves(avg_rewards, std_rewards, avg_losses, std_losses)

def plot_multi_seed_curves(avg_rewards, std_rewards, avg_losses, std_losses):
    """
    绘制多种子平均的训练曲线
    """
    plt.figure(figsize=(12, 5))

    # 设置样式
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12

    # 评估回合数（每50回合评估一次）
    episodes = np.arange(len(avg_rewards)) * 50

    # 1. 奖励曲线
    plt.subplot(1, 2, 1)

    # 平滑处理
    window = 3
    weights = np.ones(window) / window
    smooth_rewards = np.convolve(avg_rewards, weights, mode='valid')
    smooth_std = np.convolve(std_rewards, weights, mode='valid')
    x_smooth = episodes[window-1:]

    plt.plot(x_smooth, smooth_rewards, color='#2E86AB', linewidth=2.5,
             label=f'Mean Reward (n=5)')

    # 标准差阴影
    plt.fill_between(x_smooth,
                     smooth_rewards - smooth_std,
                     smooth_rewards + smooth_std,
                     alpha=0.2, color='#2E86AB',
                     label='±1 std')

    plt.title("Evaluation on Complex Maps", fontsize=14, pad=15)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Average Episode Reward", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='lower left')

    # 2. 损失曲线
    plt.subplot(1, 2, 2)

    smooth_losses = np.convolve(avg_losses, weights, mode='valid')
    smooth_loss_std = np.convolve(std_losses, weights, mode='valid')

    plt.plot(x_smooth, smooth_losses, color='#A23B72', linewidth=2,
             label='Mean Loss (n=5)')

    # 标准差阴影
    plt.fill_between(x_smooth,
                     np.maximum(0.01, smooth_losses - smooth_loss_std),
                     smooth_losses + smooth_loss_std,
                     alpha=0.15, color='#A23B72')

    plt.title("PPO Training Loss", fontsize=14, pad=15)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Loss (log scale)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.yscale('log')
    plt.ylim(bottom=0.01)

    plt.tight_layout(pad=3.0)
    plt.savefig('multi_seed_training_curve.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print("\n多种子训练曲线已保存: multi_seed_training_curve.png")

if __name__ == '__main__':
    # 导入random模块
    import random

    # 训练多个种子
    train_multiple_seeds()