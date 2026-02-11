
class UFOSSA-Mamba(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(UFOSSA-Mamba, self).__init__()
        filters = [num_channels, 64, 128, 256, 512]

        """Encoder"""
        resnet = res2net50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.cencoder1 = nn.Conv2d(filters[1] * 4, filters[1], kernel_size=1)
        self.cencoder2 = nn.Conv2d(filters[2] * 4, filters[2], kernel_size=1)
        self.cencoder3 = nn.Conv2d(filters[3] * 4, filters[3], kernel_size=1)
        self.cencoder4 = nn.Conv2d(filters[4] * 4, filters[4], kernel_size=1)

        self.dd1 = DACblockmy(filters[1])
        self.down1 = Down(filters[1], filters[4] // 4, k=8, p=8)
        self.dd2 = DACblockmy(filters[2])
        self.down2 = Down(filters[2], filters[4] // 4, k=4, p=4) 
        self.dd3 = DACblockmy(filters[3])
        self.down3 = Down(filters[3], filters[4] // 4, k=2, p=2) 
        self.dd4 = DACblockmy(filters[4])
        self.down4 = Down(filters[4], filters[4] // 4, k=1, p=1)  


        self.top_fusion = nn.Sequential(
            nn.Conv2d(filters[4], filters[4], 3, padding=1),  # 512->512
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True)
        )

        self.affm1 = EnhancedAFFM(dim=filters[1])
        self.affm2 = EnhancedAFFM(dim=filters[2])
        self.affm3 = EnhancedAFFM(dim=filters[3])
        self.affm4 = EnhancedAFFM(dim=filters[4])

        self.decoder4 = EnhancedDecoderBlock(filters[4], filters[3])  
        self.decoder3 = EnhancedDecoderBlock(filters[3], filters[2])  
        self.decoder2 = EnhancedDecoderBlock(filters[2], filters[1]) 
        self.decoder1 = DecoderBlocku(filters[1], filters[1] // 2)  

        self.aux_4 = nn.Sequential(
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
            nn.Conv2d(filters[3], 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 3, padding=1)
        )
        self.aux_3 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            nn.Conv2d(filters[2], 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 3, padding=1)
        )
        self.aux_2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(filters[1], 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 3, padding=1)
        )

        """Final output"""
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(filters[1] // 2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 3, padding=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output_dict = {}

        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x0 = self.firstrelu(x)  
        x1 = self.firstmaxpool(x0)  
        e1 = self.encoder1(x1)  
        e2 = self.encoder2(e1)  
        e3 = self.encoder3(e2)  
        e4 = self.encoder4(e3)  

        # 调整通道数
        e1 = self.cencoder1(e1)  
        e2 = self.cencoder2(e2)  
        e3 = self.cencoder3(e3)  
        e4 = self.cencoder4(e4)  


        skip1 = self.dd1(e1)  # [B, 64, 56, 56]
        ce1 = self.down1(skip1)  # [B, 128, 7, 7]

        skip2 = self.dd2(e2)  # [B, 128, 28, 28]
        ce2 = self.down2(skip2)  # [B, 128, 7, 7]

        skip3 = self.dd3(e3)  # [B, 256, 14, 14]
        ce3 = self.down3(skip3)  # [B, 128, 7, 7]

        skip4 = self.dd4(e4)  # [B, 512, 7, 7]
        ce4 = self.down4(skip4)  # [B, 128, 7, 7]

        topc = torch.cat([ce1, ce2, ce3, ce4], dim=1)  # [B, 512, 7, 7]
        topc = self.top_fusion(topc)  # [B, 512, 7, 7]

        d4 = self.affm4(topc)  # [B, 256, 14, 14]
        d4 = self.decoder4(d4, skip3)  # [B, 256, 14, 14]
        aux4 = self.aux_4(d4)

        d3 = self.affm3(d4)  # [B, 128, 28, 28]
        d3 = self.decoder3(d3, skip2)  # [B, 128, 28, 28]
        aux3 = self.aux_3(d3)

        d2 = self.affm2(d3)  # [B, 64, 56, 56]
        d2 = self.decoder2(d2, skip1)  # [B, 64, 56, 56]
        aux2 = self.aux_2(d2)

        d1 = self.affm1(d2)  # [B, 64, 56, 56]
        d1 = self.decoder1(d1)  # [B, 32, 56, 56]

        # Final upsampling to original size
        out = self.final_upsample(d1)  # [B, 32, 112, 112]
        out = self.final_conv(out)  # [B, num_classes, 112, 112]

        output_dict['output'] = self.sigmoid(out)
        output_dict['aux4'] = self.sigmoid(aux4)
        output_dict['aux3'] = self.sigmoid(aux3)
        output_dict['aux2'] = self.sigmoid(aux2)

        output_dict['output_e'] = out
        output_dict['aux4_e'] = aux4
        output_dict['aux3_e'] = aux3
        output_dict['aux2_e'] = aux2

        return output_dict