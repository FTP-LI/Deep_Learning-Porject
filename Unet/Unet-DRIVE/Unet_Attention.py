import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(torch.nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Conv, self).__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(inchannel, outchannel, 3, 1, 1),
            torch.nn.BatchNorm2d(outchannel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(outchannel, outchannel, 3, 1, 1),
            torch.nn.BatchNorm2d(outchannel),
            torch.nn.ReLU()
        )
    
    def forward(self, x):
        return self.feature(x)

class ImprovedAttentionGate(nn.Module):
    """改进的注意力门控模块 - 添加残差连接和可学习权重"""
    def __init__(self, gate_channels, skip_channels, inter_channels=None):
        super(ImprovedAttentionGate, self).__init__()
        
        if inter_channels is None:
            inter_channels = skip_channels // 2
            if inter_channels == 0:
                inter_channels = 1
                
        # 门控信号处理
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        # 跳跃连接处理
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        # 注意力系数生成 - 使用更温和的激活
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        # 可学习的权重参数 - 控制注意力强度
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始化为0.5，允许学习
        
        # 残差连接的权重
        self.beta = nn.Parameter(torch.tensor(0.8))   # 残差连接权重
        
    def forward(self, gate, skip):
        # 获取输入尺寸
        gate_size = gate.size()
        skip_size = skip.size()
        
        # 处理门控信号
        gate_conv = self.W_gate(gate)
        
        # 处理跳跃连接
        skip_conv = self.W_skip(skip)
        
        # 如果尺寸不匹配，调整门控信号尺寸
        if gate_size[2:] != skip_size[2:]:
            gate_conv = F.interpolate(gate_conv, size=skip_size[2:], mode='bilinear', align_corners=False)
        
        # 计算注意力系数
        attention = self.relu(gate_conv + skip_conv)
        attention = self.psi(attention)
        
        # 改进的注意力应用 - 添加残差连接和可学习权重
        # 使用更温和的注意力应用方式
        attention_scaled = self.alpha * attention + (1 - self.alpha) * 0.8  # 确保不会完全抑制
        attended_skip = skip * attention_scaled
        
        # 添加残差连接
        output = self.beta * attended_skip + (1 - self.beta) * skip
        
        return output

class ImprovedChannelAttention(nn.Module):
    """改进的通道注意力模块 - 降低抑制强度"""
    def __init__(self, channels, reduction=16):
        super(ImprovedChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # 使用更温和的激活函数
        self.sigmoid = nn.Sigmoid()
        
        # 可学习的权重参数
        self.gamma = nn.Parameter(torch.tensor(0.3))  # 降低通道注意力的影响
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        
        # 更温和的注意力应用
        attention_scaled = self.gamma * attention + (1 - self.gamma) * 0.9
        return x * attention_scaled

class MultiScaleAttention(nn.Module):
    """多尺度注意力模块"""
    def __init__(self, channels):
        super(MultiScaleAttention, self).__init__()
        
        # 不同尺度的卷积
        self.conv1x1 = nn.Conv2d(channels, channels // 4, 1)
        self.conv3x3 = nn.Conv2d(channels, channels // 4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(channels, channels // 4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(channels, channels // 4, 7, padding=3)
        
        # 融合层
        self.fusion = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()
        
        # 权重参数
        self.delta = nn.Parameter(torch.tensor(0.2))  # 多尺度注意力权重
        
    def forward(self, x):
        # 多尺度特征提取
        feat1 = self.conv1x1(x)
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)
        feat7 = self.conv7x7(x)
        
        # 特征融合
        multi_scale = torch.cat([feat1, feat3, feat5, feat7], dim=1)
        attention = self.sigmoid(self.bn(self.fusion(multi_scale)))
        
        # 温和的注意力应用
        attention_scaled = self.delta * attention + (1 - self.delta) * 0.85
        return x * attention_scaled

class AttentionUNet(torch.nn.Module):
    """改进的带注意力机制的UNet模型"""
    def __init__(self, inchannel, outchannel):
        super(AttentionUNet, self).__init__()
        
        # 编码器
        self.conv1 = Conv(inchannel, 64)
        self.conv2 = Conv(64, 128)
        self.conv3 = Conv(128, 256)
        self.conv4 = Conv(256, 512)
        self.conv5 = Conv(512, 1024)
        self.pool = torch.nn.MaxPool2d(2)
        
        # 改进的通道注意力模块
        self.ca1 = ImprovedChannelAttention(64)
        self.ca2 = ImprovedChannelAttention(128)
        self.ca3 = ImprovedChannelAttention(256)
        self.ca4 = ImprovedChannelAttention(512)
        
        # 多尺度注意力模块
        self.msa1 = MultiScaleAttention(64)
        self.msa2 = MultiScaleAttention(128)
        self.msa3 = MultiScaleAttention(256)
        self.msa4 = MultiScaleAttention(512)
        
        # 解码器 - 使用改进的注意力门控
        self.up1 = torch.nn.ConvTranspose2d(1024, 512, 2, 2)
        self.att1 = ImprovedAttentionGate(gate_channels=512, skip_channels=512)
        self.conv6 = Conv(1024, 512)
        
        self.up2 = torch.nn.ConvTranspose2d(512, 256, 2, 2)
        self.att2 = ImprovedAttentionGate(gate_channels=256, skip_channels=256)
        self.conv7 = Conv(512, 256)
        
        self.up3 = torch.nn.ConvTranspose2d(256, 128, 2, 2)
        self.att3 = ImprovedAttentionGate(gate_channels=128, skip_channels=128)
        self.conv8 = Conv(256, 128)
        
        self.up4 = torch.nn.ConvTranspose2d(128, 64, 2, 2)
        self.att4 = ImprovedAttentionGate(gate_channels=64, skip_channels=64)
        self.conv9 = Conv(128, 64)
        
        # 输出层
        self.conv10 = torch.nn.Conv2d(64, outchannel, 3, 1, 1)
        
        # 深度监督输出 - 权重将在训练中大幅降低
        self.aux_conv1 = nn.Conv2d(512, outchannel, 1)
        self.aux_conv2 = nn.Conv2d(256, outchannel, 1)
        
    def forward(self, x):
        # 编码器路径 - 应用改进的注意力机制
        xc1 = self.conv1(x)
        xc1_ca = self.ca1(xc1)  # 通道注意力
        xc1_att = self.msa1(xc1_ca)  # 多尺度注意力
        xp1 = self.pool(xc1_att)
        
        xc2 = self.conv2(xp1)
        xc2_ca = self.ca2(xc2)
        xc2_att = self.msa2(xc2_ca)
        xp2 = self.pool(xc2_att)
        
        xc3 = self.conv3(xp2)
        xc3_ca = self.ca3(xc3)
        xc3_att = self.msa3(xc3_ca)
        xp3 = self.pool(xc3_att)
        
        xc4 = self.conv4(xp3)
        xc4_ca = self.ca4(xc4)
        xc4_att = self.msa4(xc4_ca)
        xp4 = self.pool(xc4_att)
        
        # 瓶颈层
        xc5 = self.conv5(xp4)
        
        # 解码器路径（带改进的注意力门控）
        xu1 = self.up1(xc5)
        xc4_gated = self.att1(gate=xu1, skip=xc4_att)
        xm1 = torch.cat([xc4_gated, xu1], dim=1)
        xc6 = self.conv6(xm1)
        
        xu2 = self.up2(xc6)
        xc3_gated = self.att2(gate=xu2, skip=xc3_att)
        xm2 = torch.cat([xc3_gated, xu2], dim=1)
        xc7 = self.conv7(xm2)
        
        xu3 = self.up3(xc7)
        xc2_gated = self.att3(gate=xu3, skip=xc2_att)
        xm3 = torch.cat([xc2_gated, xu3], dim=1)
        xc8 = self.conv8(xm3)
        
        xu4 = self.up4(xc8)
        xc1_gated = self.att4(gate=xu4, skip=xc1_att)
        xm4 = torch.cat([xc1_gated, xu4], dim=1)
        xc9 = self.conv9(xm4)
        
        # 主输出
        xc10 = self.conv10(xc9)
        
        # 如果是训练模式，返回深度监督输出（权重将大幅降低）
        if self.training:
            aux1 = self.aux_conv1(xc6)
            aux2 = self.aux_conv2(xc7)
            aux1 = F.interpolate(aux1, size=x.shape[2:], mode='bilinear', align_corners=False)
            aux2 = F.interpolate(aux2, size=x.shape[2:], mode='bilinear', align_corners=False)
            return xc10, aux1, aux2
        
        return xc10

if __name__ == "__main__":
    input_tensor = torch.randn((1, 1, 512, 512))
    model = AttentionUNet(1, 1)
    
    # 测试训练模式
    model.train()
    output_train = model(input_tensor)
    print(f"训练模式输出数量: {len(output_train)}")
    print(f"主输出形状: {output_train[0].shape}")
    print(f"辅助输出1形状: {output_train[1].shape}")
    print(f"辅助输出2形状: {output_train[2].shape}")
    
    # 测试推理模式
    model.eval()
    output_eval = model(input_tensor)
    print(f"推理模式输出形状: {output_eval.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")