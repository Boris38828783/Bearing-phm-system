"""
1D-CNN 模型：用于轴承振动信号的故障分类
"""
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    一维卷积神经网络，用于振动信号故障诊断。

    结构：3 个卷积块 -> 全局平均池化 -> 全连接分类层

    参数：
        num_classes (int): 分类类别数，默认 3（正常/内圈/外圈）
        in_channels (int): 输入通道数，默认 1（单路振动信号）
    """

    def __init__(self, num_classes=3, in_channels=1):
        super().__init__()

        # ---- 卷积块 1：大卷积核捕捉宽范围的冲击特征 ----
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=1, padding=32),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)      # 长度 /4
        )

        # ---- 卷积块 2：小核提取更精细的组合特征 ----
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)      # 长度 /2
        )

        # ---- 卷积块 3：进一步抽象为高层语义 ----
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)      # 长度 /2
        )

        # ---- 全局平均池化：把每个通道压缩成 1 个数 ----
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ---- 分类头 ----
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        前向传播。

        参数：
            x (Tensor): 形状 (批大小, 通道数, 信号长度)

        返回：
            Tensor: 形状 (批大小, num_classes) 的类别得分
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.global_pool(x)      # (B, 64, 1)
        x = x.squeeze(-1)            # (B, 64)

        x = self.dropout(x)
        x = self.fc(x)               # (B, num_classes)
        return x