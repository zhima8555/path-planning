import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# 设置样式
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

def visualize_attention_weights(model, obs_dict, hidden_state, save_path='attention_visualization.png'):
    """
    可视化双重注意力机制的权重分布

    Args:
        model: CascadedDualAttentionActorCritic模型
        obs_dict: 观测字典
        hidden_state: 隐藏状态
        save_path: 保存路径
    """
    model.eval()

    # 获取中间结果
    with torch.no_grad():
        # 提取图像
        x_img = obs_dict['image']  # [1, 3, 40, 40]

        # 获取CNN特征
        cnn_features = model.dual_attention_cnn.cnn(x_img)  # [1, 64, 1, 1]

        # 获取自注意力权重
        cnn_flat = cnn_features.flatten(2).transpose(1, 2)  # [1, 1, 64]

        # 计算自注意力权重
        self_attn = model.dual_attention_cnn.self_attention
        qkv = self_attn.qkv(cnn_flat)
        B, N, _ = qkv.shape

        # 分离Q, K, V
        qkv = qkv.reshape(B, N, 3, self_attn.num_heads, self_attn.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # 计算注意力分数
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self_attn.head_dim))
        attn_weights = attn_scores.softmax(dim=-1)

        # 获取路径特征
        path_channel = x_img[:, 1:2, :, :]
        path_features = model.dual_attention_cnn.path_extractor(path_channel)

        # 计算交叉注意力权重
        cross_attn = model.dual_attention_cnn.cross_attention
        Q = cross_attn.query(path_features).unsqueeze(1)
        K = cross_attn.key(cnn_flat)

        Q = Q.view(B, 1, cross_attn.num_heads, cross_attn.head_dim).transpose(1, 2)
        K = K.view(B, N, cross_attn.num_heads, cross_attn.head_dim).transpose(1, 2)

        cross_attn_scores = (Q @ K.transpose(-2, -1)) * (1.0 / np.sqrt(cross_attn.head_dim))
        cross_attn_weights = cross_attn_scores.softmax(dim=-1)

    # 创建可视化
    fig = plt.figure(figsize=(16, 10))

    # 1. 原始观测
    ax1 = plt.subplot(2, 4, 1)
    img = x_img[0].cpu().numpy()
    # 合并三个通道用于显示
    display_img = np.zeros((40, 40, 3))
    display_img[:, :, 0] = img[0]  # 静态障碍 - 红色
    display_img[:, :, 1] = img[1]  # 路径 - 绿色
    display_img[:, :, 2] = img[2]  # 动态障碍 - 蓝色
    ax1.imshow(display_img, origin='lower')
    ax1.set_title('原始观测\n(红:障碍, 绿:路径, 蓝:动态物)')
    ax1.axis('off')

    # 2. 单独显示各通道
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(img[0], cmap='Reds', origin='lower', vmin=0, vmax=1)
    ax2.set_title('静态障碍物')
    ax2.axis('off')

    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(img[1], cmap='Greens', origin='lower', vmin=0, vmax=1)
    ax3.set_title('全局路径')
    ax3.axis('off')

    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(img[2], cmap='Blues', origin='lower', vmin=0, vmax=1)
    ax4.set_title('动态障碍物')
    ax4.axis('off')

    # 5. 自注意力权重热力图
    ax5 = plt.subplot(2, 4, 5)
    # 由于只有一个空间位置，我们显示注意力头之间的权重
    self_attn_avg = attn_weights[0].mean(dim=0).cpu().numpy()  # [num_heads, N]
    sns.heatmap(self_attn_avg, annot=True, cmap='coolwarm', center=0,
                ax=ax5, cbar_kws={'label': '注意力权重'})
    ax5.set_title(f'自注意力权重\n(多头部平均)')
    ax5.set_xlabel('特征位置')
    ax5.set_ylabel('注意力头')

    # 6. CNN特征图可视化
    ax6 = plt.subplot(2, 4, 6)
    # 获取前16个特征通道
    feat_map = cnn_features[0, :16].cpu().numpy()
    # 重排为4x4网格显示
    feat_grid = np.zeros((8, 8))
    idx = 0
    for i in range(4):
        for j in range(4):
            feat_grid[i*2:i*2+2, j*2:j*2+2] = feat_map[idx]
            idx += 1
    im = ax6.imshow(feat_grid, cmap='viridis', origin='lower')
    ax6.set_title('CNN特征图\n(前16个通道)')
    ax6.axis('off')
    plt.colorbar(im, ax=ax6, fraction=0.046)

    # 7. 交叉注意力权重
    ax7 = plt.subplot(2, 4, 7)
    cross_attn_avg = cross_attn_weights[0].mean(dim=0).cpu().numpy()
    sns.heatmap(cross_attn_avg, annot=True, cmap='YlOrRd',
                ax=ax7, cbar_kws={'label': '交叉注意力权重'})
    ax7.set_title(f'交叉注意力权重\n(路径→环境)')
    ax7.set_xlabel('环境特征位置')
    ax7.set_ylabel('注意力头')

    # 8. 注意力权重分布
    ax8 = plt.subplot(2, 4, 8)
    # 绘制所有注意力头的权重分布
    all_weights = attn_weights[0].cpu().numpy().flatten()
    ax8.hist(all_weights, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax8.set_title('自注意力权重分布')
    ax8.set_xlabel('权重值')
    ax8.set_ylabel('频次')
    ax8.grid(True, alpha=0.3)

    plt.suptitle('双重注意力机制可视化', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"注意力可视化已保存到: {save_path}")

    # 返回注意力权重用于进一步分析
    return {
        'self_attention_weights': attn_weights[0].cpu().numpy(),
        'cross_attention_weights': cross_attn_weights[0].cpu().numpy(),
        'cnn_features': cnn_features[0].cpu().numpy()
    }

def create_attention_gif(model, env, num_steps=50, save_path='attention_animation.gif'):
    """
    创建注意力权重的动画GIF

    Args:
        model: 训练好的模型
        env: 环境
        num_steps: 动画步数
        save_path: 保存路径
    """
    from PIL import Image
    import io

    obs_dict, _ = env.reset()
    hidden_state = torch.zeros(1, model.hidden_dim)

    frames = []

    for step in range(num_steps):
        # 选择动作
        with torch.no_grad():
            mu, std, value, next_hidden = model(obs_dict, hidden_state)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()

        # 获取注意力可视化
        attn_data = visualize_attention_weights(
            model,
            {'image': torch.FloatTensor(obs_dict['image']).unsqueeze(0),
             'vector': torch.FloatTensor(obs_dict['vector']).unsqueeze(0)},
            hidden_state,
            save_path=f'temp_attention_{step:03d}.png'
        )

        # 读取图像并调整大小
        img = Image.open(f'temp_attention_{step:03d}.png')
        img = img.resize((800, 600))
        frames.append(img)

        # 执行动作
        obs_dict, reward, done, _, _ = env.step(action.cpu().numpy())
        hidden_state = next_hidden

        if done:
            break

    # 保存GIF
    if frames:
        frames[0].save(save_path, format='GIF', append_images=frames[1:],
                      save_all=True, duration=200, loop=0)
        print(f"注意力动画已保存到: {save_path}")

# 测试代码
if __name__ == "__main__":
    from model import CascadedDualAttentionActorCritic
    from env import AutonomousNavEnv

    # 创建模型和环境
    model = CascadedDualAttentionActorCritic(action_dim=2)
    env = AutonomousNavEnv(map_type='simple')

    # 重置环境
    obs_dict, _ = env.reset()

    # 创建假观测
    test_obs = {
        'image': torch.randn(1, 3, 40, 40),
        'vector': torch.randn(1, 2)
    }
    hidden_state = torch.zeros(1, 256)

    # 可视化注意力
    attn_data = visualize_attention_weights(model, test_obs, hidden_state)

    print("注意力可视化完成!")