import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUnet(nn.Module):
    def __init__(self, input_channel=1, output_channel=2, init_dim=64):
        super(SimpleUnet, self).__init__()

        # エンコーダ（ダウンサンプリング）
        self.enc1 = self.double_conv(input_channel, init_dim)
        self.enc2 = self.double_conv(init_dim, init_dim * 2)
        self.enc3 = self.double_conv(init_dim * 2, init_dim * 4)
        self.enc4 = self.double_conv(init_dim * 4, init_dim * 8)
        self.enc5 = self.double_conv(init_dim * 8, init_dim * 16)

        # 最大プーリング
        self.pool = nn.MaxPool2d(2)

        # デコーダ（アップサンプリング）
        self.up4 = self.upconv(init_dim * 16, init_dim * 8)
        self.dec4 = self.double_conv(init_dim * 16, init_dim * 8)

        self.up3 = self.upconv(init_dim * 8, init_dim * 4)
        self.dec3 = self.double_conv(init_dim * 8, init_dim * 4)

        self.up2 = self.upconv(init_dim * 4, init_dim * 2)
        self.dec2 = self.double_conv(init_dim * 4, init_dim * 2)

        self.up1 = self.upconv(init_dim * 2, init_dim)
        self.dec1 = self.double_conv(init_dim * 2, init_dim)

        # 出力層（1×1 Conv）
        self.final_conv = nn.Conv2d(init_dim, output_channel, kernel_size=1)

    def forward(self, x):
        # エンコーダ
        enc1 = self.enc1(x)  
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        # デコーダ + スキップ接続
        dec4 = self.up4(enc5)
        dec4 = self.concat(dec4, enc4)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)
        dec3 = self.concat(dec3, enc3)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = self.concat(dec2, enc2)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = self.concat(dec1, enc1)
        dec1 = self.dec1(dec1)

        # 出力
        out = self.final_conv(dec1)
        return out

    def double_conv(self, in_channels, out_channels):
        """ 3x3 Conv → BatchNorm → ReLU → 3x3 Conv → BatchNorm → ReLU """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # バッチ正規化
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # バッチ正規化
            nn.LeakyReLU(0.2, True)
        )

    def upconv(self, in_channels, out_channels):
        """ 2x2 転置畳み込み（アップサンプリング） """
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def concat(self, upsampled, bypass):
        """ スキップ接続用のクロップ＆結合 """
        _, _, H, W = upsampled.shape
        bypass = F.interpolate(bypass, size=(H, W), mode='bilinear', align_corners=True)
        return torch.cat([upsampled, bypass], dim=1)

if __name__=="__main__":
    # モデルの作成例
    model = SimpleUnet(input_channel=2, output_channel=2)
    #print(model)

    input = torch.zeros(10, 2, 512, 128)

    print(model(input).shape)
