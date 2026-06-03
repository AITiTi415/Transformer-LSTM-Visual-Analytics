import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import TemporalEncoder, SpatialSceneEncoder, MASPP_1D

# ==========================================
# 变体 1：w/o Transformer (砍掉时序分支的注意力，仅保留 BiLSTM)
# ==========================================
class TemporalEncoder_NoTransformer(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super(TemporalEncoder_NoTransformer, self).__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.bilstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, 
                              num_layers=1, batch_first=True, bidirectional=True)
        # 【剥离】MultiheadAttention

    def forward(self, x):
        proj_x = self.projection(x)
        lstm_out, _ = self.bilstm(proj_x) 
        # 直接对 LSTM 输出进行时间维度均值池化
        context_vector = torch.mean(lstm_out, dim=1) 
        return context_vector

# ==========================================
# 变体 2：w/o MASPP (砍掉多尺度空间池化，仅用规则投影)
# ==========================================
class SpatialSceneEncoder_NoMASPP(nn.Module):
    def __init__(self, num_scenes=35, embed_dim=128):
        super(SpatialSceneEncoder_NoMASPP, self).__init__()
        prior_data = np.load('data/Scene_Priors.npy')
        self.prior_embedding = nn.Embedding.from_pretrained(
            torch.tensor(prior_data, dtype=torch.float32), freeze=True)
        self.rule_projection = nn.Linear(4, embed_dim)
        # 【剥离】MASPP_1D

    def forward(self, scene_id):
        priors = self.prior_embedding(scene_id)
        embeds = self.rule_projection(priors) 
        # 失去多尺度感受野，直接输出 128 维粗糙先验
        return embeds 

# ==========================================
# 打包变体主干网络
# ==========================================
class AblationPredictor(nn.Module):
    def __init__(self, ablation_type, num_abilities=8):
        super(AblationPredictor, self).__init__()
        self.ablation_type = ablation_type
        
        # 变体 3：w/o Spatial Priors (不要场景先验分支)
        if ablation_type == 'no_priors':
            self.branch_a = TemporalEncoder()
            self.branch_b = None
            self.fc_heads = nn.Linear(128, num_abilities) # 仅靠 128 维时序特征硬猜
            
        else:
            self.branch_a = TemporalEncoder_NoTransformer() if ablation_type == 'no_transformer' else TemporalEncoder()
            self.branch_b = SpatialSceneEncoder_NoMASPP() if ablation_type == 'no_maspp' else SpatialSceneEncoder()
            
            # 变体 4：w/o AFF Gate (取消自适应门控，采用暴力拼接降维)
            if ablation_type == 'no_aff':
                self.concat_proj = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
            else:
                self.aff_gate = nn.Sequential(nn.Linear(256, 128), nn.Sigmoid())
                
            self.fc_heads = nn.Linear(128, num_abilities)

    def forward(self, x_seq, scene_id):
        feat_a = self.branch_a(x_seq)
        
        if self.ablation_type == 'no_priors':
            fused_feat = feat_a
        else:
            feat_b = self.branch_b(scene_id)
            if self.ablation_type == 'no_aff':
                concat_feat = torch.cat([feat_a, feat_b], dim=1)
                fused_feat = self.concat_proj(concat_feat) # 暴力融合
            else:
                concat_feat = torch.cat([feat_a, feat_b], dim=1)
                attention_weight = self.aff_gate(concat_feat)
                fused_feat = attention_weight * feat_a + (1 - attention_weight) * feat_b # 优雅融合
                
        logits = self.fc_heads(fused_feat)
        return F.softmax(logits, dim=1)