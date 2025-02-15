import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])
        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if(Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class SimpleUnet(nn.Module):
    def __init__(self, init_dim=64, input_nc=2, output_nc=2):
        super(SimpleUnet, self).__init__()
        
        self.middle_dim = init_dim*16

        # エンコーダ（ダウンサンプリング）
        self.init_conv = nn.Conv2d(input_nc, init_dim, 1)
        
        self.convlayer1 = unet_conv(init_dim, init_dim * 2)
        self.convlayer2 = unet_conv(init_dim * 2, init_dim * 4)
        self.convlayer3 = unet_conv(init_dim * 4, init_dim * 8)
        self.convlayer4 = unet_conv(init_dim * 8, init_dim * 16)

        self.middle_layer = nn.Conv2d(self.middle_dim, self.middle_dim, kernel_size=3, stride=1, padding=1)
        
        # デコーダ（アップサンプリング）
        self.upconvlayer1 = unet_upconv(init_dim * 16, init_dim * 8)
        self.upconvlayer2 = unet_upconv(init_dim * 16, init_dim * 4)
        self.upconvlayer3 = unet_upconv(init_dim * 8, init_dim * 2)
        
        self.upconvlayer4 = unet_upconv(init_dim * 2, output_nc, True)
        
        self.output_conv = nn.Conv2d(init_dim, output_nc, 1)

    def forward(self, x):
        # エンコーダパート
        enc1 = self.init_conv(x)  # [B, init_dim, H/2, W/2]
        enc2 = self.convlayer1(enc1)  # [B, init_dim*2, H/4, W/4]
        enc3 = self.convlayer2(enc2)  # [B, init_dim*4, H/8, W/8]
        enc4 = self.convlayer3(enc3)  # [B, init_dim*8, H/16, W/16]
        enc5 = self.convlayer4(enc4)  # [B, init_dim*16, H/32, W/32]

        # デコーダパート with スキップ接続
        dec1 = self.upconvlayer1(enc5)  # [B, init_dim*8, H/16, W/16]
        dec1 = torch.cat([dec1, enc4], dim=1)  # スキップ接続（enc4とdec1を結合）

        dec2 = self.upconvlayer2(dec1)  # [B, init_dim*4, H/8, W/8]
        dec2 = torch.cat([dec2, enc3], dim=1)  # スキップ接続（enc3とdec2を結合）

        dec3 = self.upconvlayer3(dec2)  # [B, init_dim*2, H/4, W/4]
        dec3 = torch.cat([dec3, enc2], dim=1)  # スキップ接続（enc2とdec3を結合）

        dec4 = self.upconvlayer4(dec3)  # [B, output_nc, H/2, W/2]

        return dec4

