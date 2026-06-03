import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # 必须导入numpy，用来读取我们的先验矩阵！

# ==========================================
# 分支A：时序偏差编码器 (BiLSTM + Transformer)
# 职责：处理 (Batch, 94, 5) 的高频动作特征，提取因果关系与宏观瞬间权重
# ==========================================
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_heads=4):
        super(TemporalEncoder, self).__init__()
        # 1. 线性投影层：保留最原始的连续偏离特征，映射到高维隐空间
        self.projection = nn.Linear(input_dim, hidden_dim)
        
        # 2. BiLSTM编码器：双向捕捉神经肌肉微观因果顺序 (如先松油门后踩刹车)
        # 输出维度将是 hidden_dim * 2 (即 128 维)
        self.bilstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, 
                              num_layers=1, batch_first=True, bidirectional=True)
        
        # 3. Transformer Multi-Head Attention：上帝视角提取跨时间步全局权重
        # embed_dim 必须对应 BiLSTM 的输出维度 128
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads, batch_first=True)
        
    def forward(self, x):
        # x shape: (Batch, 94, 5)
        proj_x = self.projection(x) # (Batch, 94, 64)
        
        # BiLSTM 提取时序因果
        lstm_out, _ = self.bilstm(proj_x) # lstm_out shape: (Batch, 94, 128)
        
        # Transformer 注意力计算 (Self-Attention)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out) # (Batch, 94, 128)
        
        # 在时间维度(dim=1)进行平均池化，浓缩为一个长度为 128 的向量代表该段动作特征
        context_vector = torch.mean(attn_out, dim=1) # (Batch, 128)
        return context_vector


# ==========================================
# 模块组件：一维多尺度空洞空间金字塔池化 (1D-MASPP)
# 职责：多尺度感受野，评估操作与空间构型的契合度
# ==========================================
class MASPP_1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MASPP_1D, self).__init__()
        # 空洞率分别为 1, 2, 4 (提取微观、中观、宏观规则)
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.branch2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.branch3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        # 降维融合
        self.project = nn.Sequential(
            nn.Conv1d(out_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Length)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.conv_pool(self.global_pool(x))
        b4 = F.interpolate(b4, size=x.size(2), mode='nearest')
        
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.project(out)


# ==========================================
# 分支B：空间场景规则编码器 (真·先验注入版)
# 职责：将 0~34 的场景ID映射到 Word文档定义的先验规则，再进行 MASPP
# ==========================================
class SpatialSceneEncoder(nn.Module):
    def __init__(self, num_scenes=35, embed_dim=128):
        super(SpatialSceneEncoder, self).__init__()
        # 【关键修复】：真正在这里加载你生成的 Scene_Priors.npy！
        prior_data = np.load('data/Scene_Priors.npy') # shape: (35, 4)
        
        # 将其固化为不需要更新的先验 Embedding
        self.prior_embedding = nn.Embedding.from_pretrained(
            torch.tensor(prior_data, dtype=torch.float32), 
            freeze=True # 冻结它！这是你文档里的标准答案，不准网络乱改！
        )
        
        # 将 4维的先验规则 (方向, 刹车, 油门, 危险度) 翻译到 128维 隐空间
        self.rule_projection = nn.Linear(4, embed_dim)
        
        # MASPP 处理 (利用一维卷积结构进行特征提纯)
        self.maspp = MASPP_1D(in_channels=1, out_channels=1) 
        
    def forward(self, scene_id):
        # scene_id shape: (Batch,)
        
        # 1. 查表：拿出当前场景的 4 维标准答案
        priors = self.prior_embedding(scene_id) # (Batch, 4)
        
        # 2. 投影到高维
        embeds = self.rule_projection(priors) # (Batch, 128)
        
        # 3. 变形为 MASPP 所需的 (Batch, Channels, Length) -> (Batch, 1, 128)
        embeds = embeds.unsqueeze(1) 
        
        maspp_out = self.maspp(embeds) # (Batch, 1, 128)
        return maspp_out.squeeze(1) # 回归 (Batch, 128)


# ==========================================
# 单场景联合预测网络 (异步解耦主干)
# 职责：融合分支A和分支B，输出单场景下的 8 项认知能力初步预测
# ==========================================
class SingleScenePredictor(nn.Module):
    def __init__(self, num_abilities=8):
        super(SingleScenePredictor, self).__init__()
        self.branch_a = TemporalEncoder()
        self.branch_b = SpatialSceneEncoder()
        
        # 注意力特征融合模块 (AFF - 简化版门控机制)
        self.aff_gate = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.Sigmoid()
        )
        
        # 8个并行的评估头，映射到8大认知能力
        self.fc_heads = nn.Linear(128, num_abilities)
        
    def forward(self, x_seq, scene_id):
        # 分别获取两个分支的特征 (128维)
        feat_a = self.branch_a(x_seq)
        feat_b = self.branch_b(scene_id) # 此时的 feat_b 已经是带着规则密码的了！
        
        # AFF: Adaptive Feature Fusion
        concat_feat = torch.cat([feat_a, feat_b], dim=1) # (Batch, 256)
        attention_weight = self.aff_gate(concat_feat)    # (Batch, 128)
        
        # 动态加权融合
        fused_feat = attention_weight * feat_a + (1 - attention_weight) * feat_b # (Batch, 128)
        
        # 映射到 8 维能力输出
        logits = self.fc_heads(fused_feat)
        
        # 全局 Softmax 激活层进行强约束，强制约束 8 个节点占比总和恒等于 1.0
        abilities_output = F.softmax(logits, dim=1)
        
        return abilities_output


# ==========================================
# 终极画像：全局场景注意力聚合模块
# 职责：将测试人员 35 个场景的初步成绩，自适应融合成 1 个综合画像
# ==========================================
class GlobalSceneAttention(nn.Module):
    def __init__(self, num_abilities=8):
        super(GlobalSceneAttention, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(num_abilities, 16),
            nn.Tanh(),
            nn.Linear(16, 1) 
        )

    def forward(self, x_35_scenes):
        # x_35_scenes shape: (Batch, 35, 8)
        attn_scores = self.attention_net(x_35_scenes) # (Batch, 35, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # 场景级权重归一化
        
        # 特征加权求和压缩
        weighted_features = torch.sum(x_35_scenes * attn_weights, dim=1) # (Batch, 8)
        
        # 二次强约束，确保雷达图谱 100% 占比
        final_profile = F.softmax(weighted_features, dim=1)
        return final_profile, attn_weights