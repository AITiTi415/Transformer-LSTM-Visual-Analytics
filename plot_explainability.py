import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import matplotlib.ticker as ticker

# 导入你的模型
from models import SingleScenePredictor, GlobalSceneAttention

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['figure.dpi'] = 300       
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.2   
plt.rcParams['xtick.direction'] = 'in' 
plt.rcParams['ytick.direction'] = 'in'

os.makedirs('outputs', exist_ok=True)

# ==========================================
def plot_loss_curve():
    """图1：自监督拟合损失曲线 (学术版)"""
    print(">>> 正在生成: 图1 - 自监督对齐损失曲线...")
    try:
        losses = np.load('checkpoints/training_loss.npy')
    except FileNotFoundError:
        print("未找到 training_loss.npy，请先运行模型训练。")
        return

    epochs = range(1, len(losses) + 1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(epochs, losses, color='#B22222', marker='s', linestyle='-', 
            linewidth=2, markersize=6, label='Training Loss')
    
    ax.fill_between(epochs, losses, np.min(losses)*0.95, alpha=0.1, color='#B22222')
    
    ax.set_title('模型自监督伪标签对齐损失收敛曲线 (Alignment Loss)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('训练轮次 (Epoch)', fontsize=12, fontweight='bold')
    ax.set_ylabel('自监督对齐损失 (Self-Supervised Loss)', fontsize=12, fontweight='bold')
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # 强制 X 轴为整数
    
    # 隐藏右侧和顶部边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('outputs/Fig1_Training_Loss.pdf')
    plt.savefig('outputs/Fig1_Training_Loss.png')
    plt.close()

def plot_scene_attention():
    """图2：全局场景注意力热力图 (SCI 学术序列折叠版)"""
    print(">>> 正在生成: 图2 - 场景注意力热力图 (5x7 折叠矩阵版)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 提取数据
    features = np.load('data/AllData_Process.npy')
    num_subjects = features.shape[0] // 35
    person_indices = [0 + s * num_subjects for s in range(35)]
    x_seq = torch.tensor(features[person_indices], dtype=torch.float32).to(device)
    s_ids = torch.arange(35, dtype=torch.long).to(device)
    
    # 加载模型
    model = SingleScenePredictor(num_abilities=8).to(device)
    model.load_state_dict(torch.load('checkpoints/Driving_Main_Brain.pth', weights_only=True, map_location=device))
    model.eval()
    
    aggregator = GlobalSceneAttention(num_abilities=8).to(device)
    aggregator.load_state_dict(torch.load('checkpoints/Global_Attention.pth', weights_only=True, map_location=device))
    aggregator.eval()
    
    with torch.no_grad():
        scene_preds = model(x_seq, s_ids)
        _, attn_weights = aggregator(scene_preds.unsqueeze(0))
        
    attn_weights_np = attn_weights.squeeze().cpu().numpy()

    attn_matrix = attn_weights_np.reshape((5, 7))
    
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    
    # 精心设计的坐标轴标签，赋予折叠矩阵物理意义
    x_labels = [f"Step {i+1}" for i in range(7)]
    y_labels = ["Scenes 1-7", "Scenes 8-14", "Scenes 15-21", "Scenes 22-28", "Scenes 29-35"]
    
    # 使用 YlOrRd (黄橙红)，完美契合“注意力/危险程度”的直觉
    sns.heatmap(attn_matrix, cmap='YlOrRd', annot=True, fmt=".3f", 
                xticklabels=x_labels, yticklabels=y_labels,
                cbar_kws={'label': '注意力分配权重 (Attention Weight)'}, 
                linewidths=2, linecolor='white', ax=ax)
    
    ax.set_title('多尺度虚拟场景序列自适应注意力热力矩阵 (Attention Heatmap)', fontsize=15, fontweight='bold', pad=18)
    ax.set_xlabel('子序列时间步长 (Sub-sequence Step)', fontsize=12, fontweight='bold')
    ax.set_ylabel('场景序列折叠分段 (Temporal Segments)', fontsize=12, fontweight='bold')
    
    # 字体加粗，增强学术排版质感
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11, fontweight='bold', rotation=0)
    
    # 强制增加最外层的黑色边框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        
    plt.tight_layout()
    plt.savefig('outputs/Fig2_Scene_Attention.pdf')
    plt.savefig('outputs/Fig2_Scene_Attention.png', dpi=300)
    plt.close()
    print(">>>  图2 - 热力图出图完毕！")

# ==========================================
def run_shap_explainability():
    """图3：SHAP 核心多维映射矩阵 (学术版)"""
    print(">>> 正在生成: 图3 - SHAP 特征边际贡献矩阵 (需分析35个场景，请稍候)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SingleScenePredictor(num_abilities=8).to(device)
    model.load_state_dict(torch.load('checkpoints/Driving_Main_Brain.pth', weights_only=True, map_location=device))
    model.eval()
    
    features = np.load('data/AllData_Process.npy')
    
    # 背景采样 (扩容大盘基线)
    bg_indices = np.linspace(0, len(features)-1, 20, dtype=int)
    bg_tensor = torch.tensor(features[bg_indices], dtype=torch.float32).to(device)

    # 包装器
    class SHAPWrapper(torch.nn.Module):
        def __init__(self, base_model, fixed_scene_id):
            super().__init__()
            self.base_model = base_model
            self.fixed_scene_id = fixed_scene_id
        def forward(self, x):
            ids = torch.full((x.size(0),), self.fixed_scene_id, dtype=torch.long, device=x.device)
            feat_a = self.base_model.branch_a(x)
            feat_b = self.base_model.branch_b(ids)
            concat_feat = torch.cat([feat_a, feat_b], dim=1)
            attention_weight = self.base_model.aff_gate(concat_feat)
            fused_feat = attention_weight * feat_a + (1 - attention_weight) * feat_b
            return self.base_model.fc_heads(fused_feat)

    wrapper = SHAPWrapper(model, fixed_scene_id=0).to(device)
    explainer = shap.GradientExplainer(wrapper, bg_tensor)
    
    # 提取 35 个全生命周期场景
    test_tensor = torch.tensor(features[0:35], dtype=torch.float32).to(device)
    shap_values = explainer.shap_values(test_tensor)
    
    matrix_8x5 = np.zeros((8, 5))
    if isinstance(shap_values, list):
        for class_idx in range(len(shap_values)):
            val = shap_values[class_idx]
            if torch.is_tensor(val): val = val.cpu().detach().numpy()
            matrix_8x5[class_idx] = np.mean(np.abs(val), axis=(0, 1))
    else:
        if torch.is_tensor(shap_values): shap_values = shap_values.cpu().detach().numpy()
        abs_shap = np.abs(shap_values)
        dim_8_idx = [i for i, d in enumerate(abs_shap.shape) if d == 8][0]
        dim_5_idx = [i for i, d in enumerate(abs_shap.shape) if d == 5][-1]
        axes_to_mean = tuple(i for i in range(abs_shap.ndim) if i not in (dim_8_idx, dim_5_idx))
        avg_shap = np.mean(abs_shap, axis=axes_to_mean)
        if dim_8_idx > dim_5_idx: avg_shap = avg_shap.T 
        matrix_8x5 = avg_shap

    row_sums = matrix_8x5.sum(axis=1, keepdims=True)
    matrix_8x5_norm = np.divide(matrix_8x5, row_sums, out=np.zeros_like(matrix_8x5), where=row_sums!=0)

    ABILITIES = ["专注注意力", "全面注意力", "应急响应力", "记忆学习力", 
                 "执行力", "空间认知能力", "动作抑制", "冲动行为"]
    FEATURE_NAMES = ['方向盘 (Steering)', '油门 (Throttle)', '刹车 (Brake)', '静视熵 (SGE)', '转移熵 (GTE)']
                 
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    sns.heatmap(matrix_8x5_norm, cmap='YlGnBu', annot=True, fmt=".1%", 
                xticklabels=FEATURE_NAMES, yticklabels=ABILITIES,
                cbar_kws={'label': '特征相对边际贡献度 (Relative Contribution)'}, 
                linewidths=1.5, linecolor='white', ax=ax)
    
    ax.set_title('多模态特征全局边际贡献映射矩阵 (SHAP Interpretation Matrix)', fontsize=15, fontweight='bold', pad=18)
    
    ax.xaxis.tick_top()
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold', rotation=0)
    
    # 增加边框线框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        
    plt.tight_layout()
    plt.savefig('outputs/Fig3_SHAP_Matrix.pdf')
    plt.savefig('outputs/Fig3_SHAP_Matrix.png')
    plt.close()
    print(">>>  图3 - 8x5 SHAP 学术特征矩阵出图完毕！")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("开始执行出图渲染...")
    print("="*50)
    plot_loss_curve()
    plot_scene_attention()
    run_shap_explainability()
    print("\n 所有学术图片已成功生成！")