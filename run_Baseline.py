import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

# 导入数据加载模块
from dataloader import get_dataloaders
from BaselineModels import CNNLSTMPredictor, TCNPredictor
def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多GPU
    # 保证cuDNN的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 立即调用生效
set_global_seed(42)


def custom_driving_loss(predictions, inputs_seq):
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
        
        # 0. 专注注意力
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

    # 加入温度系数并进行 Softmax 强约束
    targets = F.softmax(pseudo_targets * 4.0, dim=1)
    return F.mse_loss(predictions, targets)

def evaluate_baseline_metrics(model, model_name, data_path='data/AllData_Process.npy'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    features = np.load(data_path)
    num_subjects = features.shape[0] // 35 
    
    all_mse, all_mae, all_cos, all_kl = [], [], [], []
    
    with torch.no_grad():
        for person_id in range(num_subjects):
            person_indices = [person_id + s * num_subjects for s in range(35)]
            x_seq = torch.tensor(features[person_indices], dtype=torch.float32).to(device)
            s_ids = torch.arange(35, dtype=torch.long).to(device)
            
            # 模型预测 (传入 s_ids 保持接口兼容)
            preds = model(x_seq, s_ids)
            
            # 复刻目标的伪标签生成逻辑
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
            
            # 计算学术评估指标
            mse = F.mse_loss(preds, targets).item()
            mae = F.l1_loss(preds, targets).item()
            cos_sim = F.cosine_similarity(preds, targets).mean().item()
            kl_div = F.kl_div(preds.clamp(min=1e-7).log(), targets, reduction='batchmean').item()
            
            all_mse.append(mse)
            all_mae.append(mae)
            all_cos.append(cos_sim)
            all_kl.append(kl_div)
            
    metrics = {
        "模型名称": model_name,
        "余弦相似度": round(np.mean(all_cos), 4),
        "KL散度": round(np.mean(all_kl), 4),
        "均方误差(MSE)": round(np.mean(all_mse), 4),
        "平均绝对误差(MAE)": round(np.mean(all_mae), 4)
    }
    return metrics

def run_experiments():
    data_path = 'data/AllData_Process.npy'
    print(">>> [启动] 正在加载【人员级】群体常模数据底座...")
    train_loader, _ = get_dataloaders(data_path, batch_size=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 20
    
    # 实验配置字典
    baseline_configs = {
        "Baseline A: CNN-LSTM": CNNLSTMPredictor(num_abilities=8),
        "Baseline B: TCN": TCNPredictor(num_abilities=8)
    }
    
    final_results = []
    
    for model_name, model in baseline_configs.items():
        print(f"\n==================================================")
        print(f" 开始训练对比模型: {model_name}")
        print(f"==================================================")
        
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0
            
            for batch_idx, subject_seqs in enumerate(train_loader):
                subject_seqs = subject_seqs.to(device) # shape: (Batch, 35, 94, 5)
                batch_size = subject_seqs.size(0)
                optimizer.zero_grad()
                
                scene_losses = 0.0
                # 遍历35个标准化虚拟现实场景
                for s in range(35):
                    scene_inputs = subject_seqs[:, s, :, :] # shape: (Batch, 94, 5)
                    scene_ids = torch.full((batch_size,), s, dtype=torch.long, device=device)
                    
                    # 前向传播与损失计算
                    preds = model(scene_inputs, scene_ids)
                    scene_losses += custom_driving_loss(preds, scene_inputs)
                
                scene_losses.backward()
                optimizer.step()
                total_train_loss += scene_losses.item()
                
            avg_loss = total_train_loss / len(train_loader)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] | 场景聚合Loss: {avg_loss:.4f}")
        
        # 训练完成后立即进行学术指标评测
        print(f">>> 训练完成，正在解码全量群体常模指标...")
        model_metrics = evaluate_baseline_metrics(model, model_name, data_path)
        final_results.append(model_metrics)
        
        # 保存对比模型权重
        os.makedirs('checkpoints', exist_ok=True)
        save_path = f"checkpoints/{model_name.replace(':', '_').replace(' ', '_')}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"  -> 权重已成功固化至: {save_path}")

    print("\n" + "="*70)
    print(" SCI 论文对比实验结果矩阵 (Baseline Results)")
    print("="*70)
    df_results = pd.DataFrame(final_results)
    print(df_results.to_string(index=False))
    print("="*70 + "\n")


if __name__ == "__main__":
    run_experiments()