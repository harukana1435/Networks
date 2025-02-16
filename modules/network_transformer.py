import torch
import torch.nn as nn
import torch.nn.functional as F

# CNNエンコーダ
class HRTF_CNN(nn.Module):
    def __init__(self):
        super(HRTF_CNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((512, 16))  # 時間を縮小

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # (B, 256, 512, 16)
        x = x.mean(dim=-1)  # 時間方向を平均化 (B, 256, 512)
        return x

# Transformerデコーダ
class HRTF_Transformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4):
        super(HRTF_Transformer, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_dim, 2)  # 左右のHRTF

    def forward(self, x):
        x = x.permute(2, 0, 1)  # (512, B, 256)
        x = self.transformer(x)  # (512, B, 256)
        x = self.fc(x)  # (512, B, 2)
        x = x.permute(1, 2, 0)  # (B, 2, 512)
        return x

# HRTF推定モデル
class HRTF_Model(nn.Module):
    def __init__(self):
        super(HRTF_Model, self).__init__()
        self.cnn = HRTF_CNN()
        self.transformer = HRTF_Transformer()

    def forward(self, spectrogram):
        features = self.cnn(spectrogram)  # (B, 256, 512)
        hrtf = self.transformer(features)  # (B, 2, 512)
        return hrtf

# ダミーデータ
spectrogram = torch.randn(8, 2, 512, 128)  # バイノーラルスペクトログラム

# モデルインスタンス
model = HRTF_Model()
hrtf_pred = model(spectrogram)  # (B, 2, 512)

# HRTFを時間方向にコピー
hrtf_time_expanded = hrtf_pred.unsqueeze(-1).expand(-1, -1, -1, 128)  # (B, 2, 512, 128)

print(hrtf_pred.shape)  # (8, 2, 512) -> 周波数応答のみ
print(hrtf_time_expanded.shape)  # (8, 2, 512, 128) -> 時間軸拡張版
