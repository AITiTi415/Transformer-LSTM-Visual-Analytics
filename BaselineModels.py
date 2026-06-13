import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMPredictor(nn.Module):
    def __init__(self, input_dim=5, num_abilities=8):
        super(CNNLSTMPredictor, self).__init__()
        
        # 1D-CNN 特征提取器
        # 输入维度: (Batch, Channels, Length) -> 需要在 forward 中进行转置
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # BiLSTM 时序捕捉器
        # 输入特征维度变为 64，输出 128 维 (64 * 2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, 
                            num_layers=1, batch_first=True, bidirectional=True)
        
        # 分类头映射到 8 大认知能力
        self.fc_heads = nn.Linear(128, num_abilities)

    def forward(self, x_seq, scene_id):
        # x_seq shape: (Batch, 94, 5)
        # 为了兼容 1D-CNN，转换维度为 (Batch, 5, 94)
        x_cnn_in = x_seq.transpose(1, 2) 
        
        # 提取局部卷积特征
        cnn_out = self.cnn(x_cnn_in) # shape: (Batch, 64, 94)
        
        # 转换回 RNN 需要的维度 (Batch, 94, 64)
        lstm_in = cnn_out.transpose(1, 2)
        
        # 提取全局时序特征
        lstm_out, _ = self.lstm(lstm_in) # shape: (Batch, 94, 128)
        
        # 时间维度全局平均池化
        context_vector = torch.mean(lstm_out, dim=1) # shape: (Batch, 128)
        
        # 认知能力映射与概率归一化
        logits = self.fc_heads(context_vector)
        return F.softmax(logits, dim=1)

class TCNPredictor(nn.Module):
    def __init__(self, input_dim=5, num_abilities=8):
        super(TCNPredictor, self).__init__()
        
        # TCN 扩张卷积块
        # 使用 dilation=1, 2, 4 指数级扩大感受野
        self.tcn_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Block 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Block 3
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 自适应全局平均池化，直接将时间步压缩为 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头映射到 8 大认知能力
        self.fc_heads = nn.Linear(128, num_abilities)

    def forward(self, x_seq, scene_id):
        # x_seq shape: (Batch, 94, 5)
        # 转换维度为一维卷积所需的 (Batch, Channels, Length) -> (Batch, 5, 94)
        x_tcn_in = x_seq.transpose(1, 2)
        
        # 通过 TCN 提取多尺度时序特征
        tcn_out = self.tcn_blocks(x_tcn_in) # shape: (Batch, 128, 94)
        
        # 全局池化压缩时间维度
        pooled_feat = self.global_pool(tcn_out).squeeze(2) # shape: (Batch, 128)
        
        # 认知能力映射与概率归一化
        logits = self.fc_heads(pooled_feat)
        return F.softmax(logits, dim=1)