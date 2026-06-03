import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SubjectLevelDataset(Dataset):
    def __init__(self, data_path):
        self.features = np.load(data_path) 
        self.num_scenes = 35
        # 【动态解耦】直接读取真实行数并除以35，拒绝写死 57！
        self.num_subjects = self.features.shape[0] // self.num_scenes

    def __len__(self):
        return self.num_subjects # 动态返回当前实际总人数

    def __getitem__(self, subject_idx):
        indices = [subject_idx + self.num_subjects * s for s in range(self.num_scenes)]
        
        # 取出该测试人员完整的 35 个场景
        subject_scenes = self.features[indices] # shape: (35, 94, 5)
        
        return torch.tensor(subject_scenes, dtype=torch.float32)

def get_dataloaders(data_path, batch_size=4, train_split=0.8):
    dataset = SubjectLevelDataset(data_path)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader