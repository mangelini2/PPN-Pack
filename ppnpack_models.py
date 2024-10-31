from unet_parts import *

class UNet4(nn.Module): # dual encoder unet cross-correlation with reflectie padding
    def __init__(self, n_classes, bilinear=False):
        super(UNet4, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc1 = DoubleConv(1, 64)
        self.inc2 = DoubleConv(2, 64)
        self.down11 = Down(64, 128)
        self.down12 = Down(128, 256)
        self.down13 = Down(256, 512)
        self.down21 = Down(64, 128)
        self.down22 = Down(128, 256)
        self.down23 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down14 = Down(512, 1024 // factor)
        self.down24 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024 // factor)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        xin = x[:,0,:,:].unsqueeze(1)
        x1 = self.inc1(xin)
        x2 = self.down11(x1)
        x3 = self.down12(x2)
        x4 = self.down13(x3)
        x5 = self.down14(x4)

        yin = x[:,1::,:,:]
        y1 = self.inc2(yin)
        y2 = self.down21(y1)
        y3 = self.down22(y2)
        y4 = self.down23(y3)
        y5 = self.down24(y4)

        N, C, H, W = y5.shape
        u5 = F.conv2d(input=F.pad(x5, (H//2-1,H//2,W//2-1,W//2), "reflect").reshape(1, N*C, H*2-1, W*2-1),\
                weight=y5.reshape(N*C, 1, H, W), groups=N*C).reshape(N, C, H, W)
        u5 = self.bn5(u5)
        N, C, H, W = y4.shape
        u4 = F.conv2d(input=F.pad(x4, (H//2-1,H//2,W//2-1,W//2), "reflect").reshape(1, N*C, H*2-1, W*2-1),\
                weight=y4.reshape(N*C, 1, H, W), groups=N*C).reshape(N, C, H, W)
        u4 = self.bn4(u4)
        N, C, H, W = y3.shape
        u3 = F.conv2d(input=F.pad(x3, (H//2-1,H//2,W//2-1,W//2), "reflect").reshape(1, N*C, H*2-1, W*2-1),\
                weight=y3.reshape(N*C, 1, H, W), groups=N*C).reshape(N, C, H, W)
        u3 = self.bn3(u3)
        N, C, H, W = y2.shape
        u2 = F.conv2d(input=F.pad(x2, (H//2-1,H//2,W//2-1,W//2), "reflect").reshape(1, N*C, H*2-1, W*2-1),\
                weight=y2.reshape(N*C, 1, H, W), groups=N*C).reshape(N, C, H, W)
        u2 = self.bn2(u2)
        N, C, H, W = y1.shape
        u1 = F.conv2d(input=F.pad(x1, (H//2-1,H//2,W//2-1,W//2), "reflect").reshape(1, N*C, H*2-1, W*2-1),\
                weight=y1.reshape(N*C, 1, H, W), groups=N*C).reshape(N, C, H, W)
        u1 = self.bn1(u1)

        x = self.up1(u5, u4)
        x = self.up2(x, u3)
        x = self.up3(x, u2)
        x = self.up4(x, u1)

        logits = self.outc(x)
        return logits

class SimpleCrossCorr(nn.Module):
    def __init__(self, n_classes, in_channels=1, out_channels=32):
        super(SimpleCrossCorr, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = out_channels
        self.n_classes = n_classes

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels//2, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            OutConv(mid_channels, n_classes)
        )
        self.unet = UNet(self.out_channels*2, self.n_classes)

        self.unet2 = UNet2(self.n_classes)
        self.inc = DoubleConv((self.out_channels+1)*2, 64)
        self.down1 = Down(64, 128)
        self.up4 = Up(128+64, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, inputs): #input_bin, input_obj, masks):
        
        input_bin = inputs[:,0,...].unsqueeze(1)
        N = input_bin.shape[0]
        BH = input_bin.shape[2]
        BW = input_bin.shape[3]
        input_bin = F.pad(input_bin, (0,BW,0,BH), "constant", 0)
        return self.unet2(input_bin, inputs[:,1::,...])
