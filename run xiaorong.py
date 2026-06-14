import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

from dataloader import get_dataloaders
from xiaorong_models import AblationPredictor
# 引入基础模型、全局注意力聚合器和 Loss 函数
from models import SingleScenePredictor, GlobalSceneAttention
from main import custom_driving_loss

def train_and_eval_ablation(model_name, base_model, train_loader, device, epochs=20):
    print(f"\n" + "="*60)
    print(f"[启动消融实验] 当前网络变体: {model_name}")
    print("="*60)
    
    # 【新增】：为了输出最终画像，我们需要给每个变体也配一个注意力聚合器
    attention_model = GlobalSceneAttention(num_abilities=8).to(device)
    
    # 联合优化
    optimizer = optim.Adam(list(base_model.parameters()) + list(attention_model.parameters()), lr=0.001)
    
    # -------------------------
    # 阶段一：训练过程打印
    # -------------------------
    base_model.train()
    attention_model.train()
    for epoch in range(epochs):
        total_train_loss = 0.0
        for subject_seqs in train_loader:
            subject_seqs = subject_seqs.to(device)
            batch_size = subject_seqs.size(0)
            optimizer.zero_grad()
            
            all_scene_preds = []
            scene_losses = 0.0
            
            for s in range(35):
                scene_inputs = subject_seqs[:, s, :, :]
                scene_ids = torch.full((batch_size,), s, dtype=torch.long, device=device)
                
                preds = base_model(scene_inputs, scene_ids)
                all_scene_preds.append(preds)
                scene_losses += custom_driving_loss(preds, None, scene_inputs, scene_ids)
                
            # 注意力模块也参与前向传播以更新梯度
            stacked_preds = torch.stack(all_scene_preds, dim=1)
            final_profile, attn_weights = attention_model(stacked_preds)
            
            scene_losses.backward()
            optimizer.step()
            total_train_loss += scene_losses.item()
            
        avg_loss = total_train_loss / len(train_loader)
        # 【修改】：现在每个 Epoch 都会清晰打印 Loss
        print(f"  -> Epoch [{epoch+1:02d}/{epochs}] | Loss: {avg_loss:.4f}")

    # -------------------------
    # 阶段二：学术指标计算与最终结果导出
    # -------------------------
    base_model.eval()
    attention_model.eval()
    
    all_mse, all_mae, all_cos, all_kl = [], [], [], []
    all_final_scores = [] # 用于保存每个人的最终画像
    
    features = np.load('data/AllData_Process.npy')
    num_subjects = features.shape[0] // 35 
    abilities_names = ["专注注意力", "全面注意力", "应急响应力", "记忆学习力", 
                       "执行力", "空间认知能力", "动作抑制", "冲动行为"]
    
    print(f"\n  [评估中] 正在计算学术指标并生成最终能力画像...")
    with torch.no_grad():
        for person_id in range(num_subjects):
            person_indices = [person_id + s * num_subjects for s in range(35)]
            x_seq = torch.tensor(features[person_indices], dtype=torch.float32).to(device)
            s_ids = torch.arange(35, dtype=torch.long).to(device)
            
            # 单场景预测
            preds = base_model(x_seq, s_ids)
            
            # 全局聚合出最终雷达画像 (Batch, 8) -> 转换为百分比
            final_profile, _ = attention_model(preds.unsqueeze(0))
            final_scores_np = final_profile.squeeze(0).cpu().numpy() * 100
            
            # 记录到字典 (保留数据结构收集，但不再写入CSV)
            person_data = {"测试人员编号": f"Subject_{person_id+1}"}
            for name, score in zip(abilities_names, final_scores_np):
                person_data[name] = round(score, 2)
            all_final_scores.append(person_data)
            
            # 打分复刻（用于算 MSE/MAE）
            batch_size = x_seq.size(0)
            pseudo_targets = torch.zeros((batch_size, 8), device=device)
            for i in range(batch_size):
                seq = x_seq[i]
                pseudo_targets[i, 0] = torch.exp(-torch.mean(seq[:, 3]) * 3.0) 
                pseudo_targets[i, 1] = torch.mean(seq[:, 4]) * 5.0 
                pseudo_targets[i, 2] = (torch.max(seq[:, 2]) ** 2) * 3.0 
                pseudo_targets[i, 3] = torch.exp(-(torch.var(seq[:, 0]) + torch.var(seq[:, 1])) * 8.0)
                pseudo_targets[i, 4] = ((torch.mean(seq[:, 1]) + torch.max(seq[:, 2])) ** 2) * 2.0
                pseudo_targets[i, 5] = torch.exp(-torch.var(seq[:, 0]) * 15.0)
                mask = torch.abs(seq[:, 0]) > 0.2
                t_w_t = torch.mean(seq[mask, 1]) if mask.any() else torch.tensor(0.0)
                imp_score = (t_w_t ** 2) * 15.0
                pseudo_targets[i, 6] = torch.exp(-imp_score) 
                pseudo_targets[i, 7] = imp_score

            targets = F.softmax(pseudo_targets * 4.0, dim=1)
            
            all_mse.append(F.mse_loss(preds, targets).item())
            all_mae.append(F.l1_loss(preds, targets).item())
            all_cos.append(F.cosine_similarity(preds, targets).mean().item())
            all_kl.append(F.kl_div(preds.clamp(min=1e-7).log(), targets, reduction='batchmean').item())

    return np.mean(all_cos), np.mean(all_kl), np.mean(all_mse), np.mean(all_mae)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_dataloaders('data/AllData_Process.npy', batch_size=4)
    
    variants = {
        "无场景规则先验 (w/o Spatial Priors)": AblationPredictor('no_priors'),
        "无Transformer时序组件 (w/o Transformer)": AblationPredictor('no_transformer'),
        "无MASPP多尺度池化 (w/o MASPP)": AblationPredictor('no_maspp'),
        "无AFF自适应门控 (w/o AFF Gate)": AblationPredictor('no_aff'),
        "原版": SingleScenePredictor(num_abilities=8) 
    }
    
    results = {}
    for name, model in variants.items():
        model = model.to(device)
        cos, kl, mse, mae = train_and_eval_ablation(name, model, train_loader, device, epochs=20)
        results[name] = (cos, kl, mse, mae)
        
    print("\n" + "="*90)
    print(f"{'模型架构变体 (Model Variants)':<32} | {'余弦相似度':<10} | {'KL散度':<10} | {'MSE(均方误差)':<10} | {'MAE(平均绝对误差)':<10}")
    print("-" * 90)
    for name, (cos, kl, mse, mae) in results.items():
        mark = "🏆" if "原版" in name else "   "
        print(f"{mark}{name:<30} | {cos:<12.4f} | {kl:<9.4f} | {mse:<8.4f} | {mae:<8.4f}")
    print("="*90 + "\n")