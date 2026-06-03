import os
# 【必加】解决 Anaconda 环境下 Intel OpenMP 库重复加载报错的问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

# 加载模块和双分支大脑模型
from dataloader import get_dataloaders
from models import SingleScenePredictor, GlobalSceneAttention

def custom_driving_loss(predictions, targets_dummy, inputs_seq, scene_ids):
    """
    inputs_seq 维度: (Batch, 94, 5) -> [0:方向盘, 1:油门, 2:刹车, 3:SGE, 4:GTE]
    """
    batch_size = inputs_seq.size(0)
    pseudo_targets = torch.zeros((batch_size, 8), device=inputs_seq.device)
    
    for i in range(batch_size):
        seq = inputs_seq[i]
        
        # 提取统计特征
        steer_var = torch.var(seq[:, 0])
        throttle_var = torch.var(seq[:, 1])
        throttle_mean = torch.mean(seq[:, 1])
        brake_max = torch.max(seq[:, 2])
        sge_mean = torch.mean(seq[:, 3])  
        gte_mean = torch.mean(seq[:, 4])  
        
        # 0. 专注注意力 (SGE越高，注视越乱，指数级扣分)
        pseudo_targets[i, 0] = torch.exp(-sge_mean * 3.0) 
        
        # 1. 全面注意力 
        pseudo_targets[i, 1] = gte_mean * 5.0 
        
        # 2. 应急响应力 
        pseudo_targets[i, 2] = (brake_max ** 2) * 3.0 
        
        # 3. 记忆学习力 
        pseudo_targets[i, 3] = torch.exp(-(steer_var + throttle_var) * 8.0)
        
        # 4. 执行力 
        pseudo_targets[i, 4] = ((throttle_mean + brake_max) ** 2) * 2.0
        
        # 5. 空间认知能力 
        pseudo_targets[i, 5] = torch.exp(-steer_var * 15.0)
        
        # 6 & 7: 动作抑制 vs 冲动行为
        uncentered_mask = torch.abs(seq[:, 0]) > 0.2
        if uncentered_mask.any():
            throttle_while_turning = torch.mean(seq[uncentered_mask, 1])
        else:
            throttle_while_turning = torch.tensor(0.0)
            
        impulse_score = (throttle_while_turning ** 2) * 15.0
        inhibition_score = torch.exp(-impulse_score) 
        
        pseudo_targets[i, 6] = inhibition_score
        pseudo_targets[i, 7] = impulse_score

    # 加入温度系数 (* 4.0)
    pseudo_targets = F.softmax(pseudo_targets * 4.0, dim=1)
    
    # 均方误差逼迫模型学习这套极端的评分标准
    loss = F.mse_loss(predictions, pseudo_targets)
    return loss


def train_model():
    print(">>> [启动] 正在加载【人员级】多模态驾驶数据底座...")
    train_loader, val_loader = get_dataloaders('data/AllData_Process.npy', batch_size=4)
    print(f"数据加载完毕。训练批次: {len(train_loader)}，验证批次: {len(val_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_model = SingleScenePredictor(num_abilities=8).to(device)
    attention_model = GlobalSceneAttention(num_abilities=8).to(device)
    
    optimizer = optim.Adam(list(base_model.parameters()) + list(attention_model.parameters()), lr=0.001)
    epochs = 20
    os.makedirs('checkpoints', exist_ok=True)

    # 初始化一个空列表，用来记录每一个 Epoch 的 Loss
    history_losses = [] 

    print(">>> [开始] 重新训练网络...")
    for epoch in range(epochs):
        base_model.train()
        attention_model.train()
        total_train_loss = 0.0
        
        for batch_idx, subject_seqs in enumerate(train_loader):
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
            
            stacked_preds = torch.stack(all_scene_preds, dim=1)
            final_profile, attn_weights = attention_model(stacked_preds)
            
            scene_losses.backward()
            optimizer.step()
            total_train_loss += scene_losses.item()
            
        avg_loss = total_train_loss / len(train_loader)
        
        # 【新增 2/3】：把算好的平均 loss 塞进列表里保存
        history_losses.append(avg_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    torch.save(base_model.state_dict(), 'checkpoints/Driving_Main_Brain.pth')
    torch.save(attention_model.state_dict(), 'checkpoints/Global_Attention.pth')
    
    # 【新增 3/3】：训练彻底结束后，把整个 loss 列表保存为 .npy 数据文件！
    np.save('checkpoints/training_loss.npy', np.array(history_losses))
    
    print("\n[完成] 模型训练结束，权重和 Loss 曲线数据已覆盖保存。")

def generate_radar_chart(person_id=0, verbose=True):
    if verbose:
        print(f"\n>>> [画像生成] 正在提取测试人员 {person_id} 的全局认知图谱...")
        
    features = np.load('data/AllData_Process.npy')

    num_subjects = features.shape[0] // 35 
    
    person_indices = [person_id + s * num_subjects for s in range(35)]
    
    x_seq = torch.tensor(features[person_indices], dtype=torch.float32) 
    s_ids = torch.arange(35, dtype=torch.long)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SingleScenePredictor(num_abilities=8).to(device)
    model.load_state_dict(torch.load('checkpoints/Driving_Main_Brain.pth', weights_only=True))
    model.eval()
    
    aggregator = GlobalSceneAttention(num_abilities=8).to(device)
    aggregator.load_state_dict(torch.load('checkpoints/Global_Attention.pth', weights_only=True))
    aggregator.eval()
    
    abilities_names = ["专注注意力", "全面注意力", "应急响应力", "记忆学习力", 
                       "执行力", "空间认知能力", "动作抑制", "冲动行为"]
                       
    with torch.no_grad():
        x_seq, s_ids = x_seq.to(device), s_ids.to(device)
        scene_preds = model(x_seq, s_ids) 
        final_profile, attn_weights = aggregator(scene_preds.unsqueeze(0))
        
    final_scores = final_profile.squeeze(0).cpu().numpy() * 100
    
    if verbose:
        print("\n========= 最终雷达图谱占比 (总计 100%) =========")
        for name, score in zip(abilities_names, final_scores):
            print(f" {name}: \t {score:.2f}%")
        print("===============================================\n")
    
    return final_scores, attn_weights


def evaluate_model_metrics():
    print("\n>>>正在计算模型核心指标 (KL散度、余弦相似度、MSE、MAE)")
    features = np.load('data/AllData_Process.npy')
    num_subjects = features.shape[0] // 35 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SingleScenePredictor(num_abilities=8).to(device)
    model.load_state_dict(torch.load('checkpoints/Driving_Main_Brain.pth', map_location=device, weights_only=True))
    model.eval()
    
    all_mse, all_mae, all_cos, all_kl = [], [], [], []
    
    with torch.no_grad():
        for person_id in range(num_subjects):
            person_indices = [person_id + s * num_subjects for s in range(35)]
            x_seq = torch.tensor(features[person_indices], dtype=torch.float32).to(device)
            s_ids = torch.arange(35, dtype=torch.long).to(device)
            
            # 模型预测
            preds = model(x_seq, s_ids)
            
            # 内部复刻提取打分规则，严防污染外部 Loss 代码
            batch_size = x_seq.size(0)
            pseudo_targets = torch.zeros((batch_size, 8), device=device)
            for i in range(batch_size):
                seq = x_seq[i]
                steer_var = torch.var(seq[:, 0])
                throttle_var = torch.var(seq[:, 1])
                throttle_mean = torch.mean(seq[:, 1])
                brake_max = torch.max(seq[:, 2])
                sge_mean = torch.mean(seq[:, 3])  
                gte_mean = torch.mean(seq[:, 4])  
                
                pseudo_targets[i, 0] = torch.exp(-sge_mean * 3.0) 
                pseudo_targets[i, 1] = gte_mean * 5.0 
                pseudo_targets[i, 2] = (brake_max ** 2) * 3.0 
                pseudo_targets[i, 3] = torch.exp(-(steer_var + throttle_var) * 8.0)
                pseudo_targets[i, 4] = ((throttle_mean + brake_max) ** 2) * 2.0
                pseudo_targets[i, 5] = torch.exp(-steer_var * 15.0)
                
                uncentered_mask = torch.abs(seq[:, 0]) > 0.2
                if uncentered_mask.any():
                    throttle_while_turning = torch.mean(seq[uncentered_mask, 1])
                else:
                    throttle_while_turning = torch.tensor(0.0)
                    
                impulse_score = (throttle_while_turning ** 2) * 15.0
                inhibition_score = torch.exp(-impulse_score) 
                
                pseudo_targets[i, 6] = inhibition_score
                pseudo_targets[i, 7] = impulse_score

            targets = F.softmax(pseudo_targets * 4.0, dim=1)
            
            # 核心学术指标计算
            mse = F.mse_loss(preds, targets).item()
            mae = F.l1_loss(preds, targets).item()
            cos_sim = F.cosine_similarity(preds, targets).mean().item()
            kl_div = F.kl_div(preds.clamp(min=1e-7).log(), targets, reduction='batchmean').item()
            
            all_mse.append(mse)
            all_mae.append(mae)
            all_cos.append(cos_sim)
            all_kl.append(kl_div)
            
    print("\n" + "="*50)
    print("学术评估指标")
    print("="*50)
    print(f"  [1] 余弦相似度 (Cosine Similarity) : {np.mean(all_cos):.4f}")
    print(f"  [2] KL 散度 (KL Divergence)       : {np.mean(all_kl):.4f}")
    print(f"  [3] 均方误差 (MSE)                : {np.mean(all_mse):.4f}")
    print(f"  [4] 平均绝对误差 (MAE)            : {np.mean(all_mae):.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # 让模型学习新的打分机制
    train_model() 
    
    evaluate_model_metrics()
    
    try:
        base_features = np.load('data/AllData_Process.npy')
        total_subjects = base_features.shape[0] // 35
    except FileNotFoundError:
        print(" 错误：未找到 data/AllData_Process.npy！请确保路径正确。")
        exit()
        
    print(f"\n>>> [批量生成] 正在提取全部 {total_subjects} 名测试人员的雷达图谱数据，请稍候...")
    all_scores = []
    abilities_names = ["专注注意力", "全面注意力", "应急响应力", "记忆学习力", 
                       "执行力", "空间认知能力", "动作抑制", "冲动行为"]
                       
    # 动态循环当前总人数
    for i in range(total_subjects):
        scores, _ = generate_radar_chart(person_id=i, verbose=False)
        person_data = {"测试人员编号": f"测试人员_{i+1}"}
        for name, score in zip(abilities_names, scores):
            person_data[name] = round(score, 2) 
        all_scores.append(person_data)
        
        # 进度条打印也改成动态适配
        if (i + 1) % 10 == 0 or i == total_subjects - 1:
            print(f"  -> 已完成处理: {i + 1} / {total_subjects} 人")
            
    df = pd.DataFrame(all_scores)
    output_path = "能力占比分析.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')