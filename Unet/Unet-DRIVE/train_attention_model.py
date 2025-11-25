import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader, Dataset
from Unet import UNet
from Unet_Attention import AttentionUNet
import copy
from tqdm import tqdm
import os
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import gaussian_filter, map_coordinates
import cv2

# 设置matplotlib中文字体，解决中文乱码问题
import platform
import warnings
import matplotlib.font_manager as fm

# 抑制所有matplotlib相关的警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
warnings.filterwarnings('ignore', message='.*does not have a glyph.*')
warnings.filterwarnings('ignore', message='.*substituting with a dummy symbol.*')

# 抑制字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 尝试设置中文字体，如果失败则使用英文
try:
    # Windows系统字体设置
    if platform.system() == 'Windows':
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    elif platform.system() == 'Darwin':  # macOS
        matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans GB']
    else:  # Linux
        matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    
    # 关键：强制禁用unicode负号，使用ASCII负号
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 10
    
    # 额外设置以避免字体警告
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'  # 数学文本字体
    matplotlib.rcParams['mathtext.default'] = 'regular'     # 数学文本样式
    
    # 测试中文字体是否可用
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, '测试', fontsize=12)
    plt.close(fig)
    
    USE_CHINESE = True
    print("✓ 中文字体设置成功")
    
except Exception as e:
    # 如果中文字体设置失败，使用英文标签
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
    matplotlib.rcParams['mathtext.default'] = 'regular'
    USE_CHINESE = False
    print("⚠ 中文字体不可用，将使用英文标签")

# 额外的警告抑制
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# 强制重建字体缓存
try:
    fm._rebuild()
except:
    pass

# 更强制性的中文字体设置
USE_CHINESE = False

try:
    # 清除现有字体设置
    matplotlib.rcdefaults()
    
    # 根据系统设置字体
    if platform.system() == 'Windows':
        # Windows系统常见中文字体
        font_candidates = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
    elif platform.system() == 'Darwin':  # macOS
        font_candidates = ['Arial Unicode MS', 'Hiragino Sans GB', 'PingFang SC']
    else:  # Linux
        font_candidates = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    
    # 检查可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_font = None
    
    for font in font_candidates:
        if font in available_fonts:
            chinese_font = font
            break
    
    if chinese_font:
        # 设置中文字体
        matplotlib.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['font.size'] = 10
        
        # 强制设置所有相关参数
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['axes.titlesize'] = 14
        matplotlib.rcParams['axes.labelsize'] = 12
        matplotlib.rcParams['legend.fontsize'] = 12
        
        # 测试中文显示
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '测试中文', fontsize=12, ha='center')
        ax.set_title('测试标题')
        plt.close(fig)
        
        USE_CHINESE = True
        print(f"✓ 成功设置中文字体: {chinese_font}")
    else:
        raise Exception("未找到可用的中文字体")
        
except Exception as e:
    # 如果中文字体设置失败，使用英文
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 10
    USE_CHINESE = False
    print(f"⚠ 中文字体设置失败: {str(e)}")
    print("将使用英文标签显示图表")

# 额外的字体缓存清理
try:
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    if USE_CHINESE and platform.system() == 'Windows':
        plt.rcParams['font.sans-serif'] = [chinese_font if 'chinese_font' in locals() else 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# ==================== 训练配置参数 ====================
# 可直接修改的训练参数
# 修改训练配置参数
TOTAL_EPOCHS = 100          # 总训练轮次
BATCH_SIZE = 1              # 批次大小
LEARNING_RATE = 0.0005      # 降低学习率从0.001到0.0005
WEIGHT_DECAY = 1e-4         # 权重衰减
EARLY_STOP_PATIENCE = 20    # 增加早停耐心值从15到20
LR_SCHEDULER_PATIENCE = 8   # 减少学习率调度耐心值从10到8
PRINT_FREQ = 5              # 增加打印频率

PRINT_FREQ = 10             # 打印频率（每多少轮打印一次学习率）

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 使用绝对路径加载数据
traindata = np.load(os.path.join(script_dir, "处理好的数据集", "trainingdataset.npy"), allow_pickle=True)
testdata = np.load(os.path.join(script_dir, "处理好的数据集", "testdataset.npy"), allow_pickle=True)

print(f"训练数据集大小: {len(traindata)}")
print(f"测试数据集大小: {len(testdata)}")
print(f"总训练轮次: {TOTAL_EPOCHS}")
print(f"批次大小: {BATCH_SIZE}")

# 损失函数定义
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.3, bce_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal + self.bce_weight * bce

# 数据增强类
class DataAugmentation:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ])
    
    def __call__(self, image, mask):
        # 应用相同的变换到图像和掩码
        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        image = self.transforms(image)
        torch.manual_seed(seed)
        mask = self.transforms(mask)
        return image, mask

# 数据库加载
class Dataset(Dataset):
    def __init__(self, data, augment=False):
        self.len = len(data)
        self.x_data = []
        self.y_data = []
        self.augment = augment
        self.augmentation = DataAugmentation() if augment else None
        
        for item in data:
            x, y = item[0], item[1]
            # 确保数据类型正确
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype(np.float32))
            else:
                x = torch.tensor(x, dtype=torch.float32)
                
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y.astype(np.float32))
            else:
                y = torch.tensor(y, dtype=torch.float32)
                
            self.x_data.append(x)
            self.y_data.append(y)

    def __getitem__(self, index):
        x, y = self.x_data[index], self.y_data[index]
        
        # 应用数据增强（仅训练时）
        if self.augment and self.augmentation:
            x, y = self.augmentation(x, y)
            
        return x, y

    def __len__(self):
        return self.len

# 数据库dataloader
Train_dataset = Dataset(traindata, augment=True)  # 训练时使用数据增强
Test_dataset = Dataset(testdata, augment=False)   # 测试时不使用数据增强
dataloader = DataLoader(Train_dataset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(Test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 训练设备选择GPU还是CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 模型初始化
model = AttentionUNet(1, 1)  # 输入1通道，输出1通道
model.to(device)

# 损失函数选择 - 使用组合损失函数
criterion = CombinedLoss()
criterion.to(device)

# 优化器和学习率调度器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATIENCE, verbose=True, min_lr=1e-6)

train_loss = []
test_loss = []

# 训练函数
# 记录实际学习率变化
lr_history = []  # 添加学习率历史记录

# 修改第300-320行的训练函数
def train(epoch, total_epochs):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    train_bar = tqdm(dataloader, desc=f'第{epoch:2d}/{total_epochs}轮-训练', 
                     ncols=100, leave=True)
    
    for batch_idx, data in enumerate(train_bar):
        datavalue, datalabel = data
        datavalue, datalabel = datavalue.to(device), datalabel.to(device)
        
        optimizer.zero_grad()
        
        # 获取模型输出
        outputs = model(datavalue)
        
        # 处理深度监督输出
        if isinstance(outputs, tuple):  # 训练模式，有深度监督
            main_output, aux1, aux2 = outputs
            
            # 计算主损失
            main_loss = criterion(main_output, datalabel)
            
            # 计算辅助损失
            aux_loss1 = criterion(aux1, datalabel) * 0.4
            aux_loss2 = criterion(aux2, datalabel) * 0.2
            
            # 总损失
            loss = main_loss + aux_loss1 + aux_loss2
        else:  # 推理模式
            # 在train函数中，将深度监督权重大幅降低
            if isinstance(outputs, tuple):
                main_output, aux1, aux2 = outputs
                main_loss = criterion(main_output, labels)
                aux_loss1 = criterion(aux1, labels)
                aux_loss2 = criterion(aux2, labels)
                # 大幅降低辅助损失权重，避免过度约束
                loss = main_loss + 0.1 * aux_loss1 + 0.05 * aux_loss2  # 从0.4和0.2降低到0.1和0.05
            else:
                loss = criterion(outputs, labels)
        
        # 检查损失是否异常
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 10.0:
            print(f"⚠️ 检测到异常损失值: {loss.item():.6f}，跳过此批次")
            continue
            
        loss.backward()
        
        # 更严格的梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        total_loss += loss.item()
        
        # 更新进度条
        train_bar.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'Avg': f'{total_loss/(batch_idx+1):.6f}'
        })
    
    avg_loss = total_loss / num_batches
    train_loss.append(avg_loss)
    
    # 记录学习率
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"第{epoch:3d}轮 - 训练损失: {avg_loss:.6f}, 学习率: {current_lr:.2e}")
    
    return avg_loss

# 测试函数
# 同样需要修改测试函数来处理深度监督输出
def test(epoch, total_epochs):
    model.eval()
    total_loss = 0
    num_batches = len(testloader)
    
    test_bar = tqdm(testloader, desc=f'第{epoch:2d}/{total_epochs}轮-测试', 
                    ncols=100, leave=True)
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_bar):
            datavalue, datalabel = data
            datavalue, datalabel = datavalue.to(device), datalabel.to(device)
            
            # 获取模型输出
            outputs = model(datavalue)
            
            # 测试时只使用主输出
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # 只取主输出
            
            loss = criterion(outputs, datalabel)
            total_loss += loss.item()
            
            # 更新进度条
            test_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg': f'{total_loss/(batch_idx+1):.6f}'
            })
    
    avg_loss = total_loss / num_batches
    test_loss.append(avg_loss)
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"第{epoch:3d}轮 - 测试损失: {avg_loss:.6f}")
    
    return avg_loss

# 早停机制
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

bestmodel = None
bestepoch = None
bestloss = np.inf
early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)

# 训练循环
print("="*50)
print("开始训练...")
print("="*50)

# 使用配置的总轮次进行训练
for epoch in tqdm(range(1, TOTAL_EPOCHS + 1), desc="总体训练进度", ncols=100):
    train_loss_epoch = train(epoch, TOTAL_EPOCHS)
    test_loss_epoch = test(epoch, TOTAL_EPOCHS)
    
    # 学习率调度
    scheduler.step(test_loss_epoch)
    
    # 保存最佳模型
    if test_loss_epoch < bestloss:
        bestloss = test_loss_epoch
        bestepoch = epoch
        bestmodel = copy.deepcopy(model)
        print(f"✓ 新的最佳模型保存在第 {epoch} 轮，损失: {bestloss:.6f}")
    
    # 早停检查
    if early_stopping(test_loss_epoch):
        print(f"\n⚠ 早停触发，在第 {epoch} 轮停止训练")
        break
    
    # 定期打印当前学习率
    if epoch % PRINT_FREQ == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📊 第{epoch}轮 - 当前学习率: {current_lr:.2e}")
    
    print("-" * 50)

print("\n" + "="*50)
print(f"训练完成！最佳轮次: {bestepoch}, 最佳损失: {bestloss:.6f}")
print("="*50)

# 保存模型
print("正在保存模型...")
torch.save(model.state_dict(), os.path.join(script_dir, "训练好的模型", "lastmodel.pt"))
if bestmodel is not None:
    torch.save(bestmodel.state_dict(), os.path.join(script_dir, "训练好的模型", "bestmodel.pt"))
print("模型保存完成！")

# 保存训练结果图（根据字体可用性选择语言）
print("正在生成训练结果图...")
plt.figure(figsize=(14, 10))

# 根据字体可用性设置标签
if USE_CHINESE:
    train_label = '训练损失'
    test_label = '测试损失'
    xlabel_1 = '训练轮次'
    ylabel_1 = '损失值'
    title_1 = '训练和测试损失曲线'
    best_label = f'最佳轮次: {bestepoch}' if bestepoch else ''
    lr_label = '学习率'
    xlabel_2 = '训练轮次'
    ylabel_2 = '学习率'
    title_2 = '学习率调度曲线'
else:
    train_label = 'Training Loss'
    test_label = 'Validation Loss'
    xlabel_1 = 'Epoch'
    ylabel_1 = 'Loss'
    title_1 = 'Training and Validation Loss Curves'
    best_label = f'Best Epoch: {bestepoch}' if bestepoch else ''
    lr_label = 'Learning Rate'
    xlabel_2 = 'Epoch'
    ylabel_2 = 'Learning Rate'
    title_2 = 'Learning Rate Schedule'

# 第一个子图：损失曲线
plt.subplot(2, 1, 1)
plt.plot(range(1, len(train_loss) + 1), train_loss, label=train_label, color='blue', linewidth=2)
plt.plot(range(1, len(test_loss) + 1), test_loss, label=test_label, color='red', linewidth=2)
plt.legend(fontsize=12)
plt.xlabel(xlabel_1, fontsize=12)
plt.ylabel(ylabel_1, fontsize=12)
plt.title(title_1, fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 标记最佳点
if bestepoch is not None:
    plt.axvline(x=bestepoch, color='green', linestyle='--', alpha=0.7, label=best_label)
    plt.legend(fontsize=12)

# 第二个子图：学习率曲线
plt.subplot(2, 1, 2)
if len(lr_history) > 0:
    plt.plot(range(1, len(lr_history) + 1), lr_history, label=lr_label, color='orange', linewidth=2)
else:
    # 备用显示
    lr_values = [LEARNING_RATE] * len(train_loss)
    plt.plot(range(1, len(lr_values) + 1), lr_values, label=lr_label, color='orange', linewidth=2)

plt.xlabel(xlabel_2, fontsize=12)
plt.ylabel(ylabel_2, fontsize=12)
plt.title(title_2, fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.legend(fontsize=12)

# 添加损失范围限制显示
plt.subplot(2, 1, 1)
# 限制y轴范围以更好地显示正常损失变化
max_loss = max(max(train_loss), max(test_loss))
if max_loss > 5.0:  # 如果最大损失过大，限制显示范围
    plt.ylim(0, min(max_loss, 2.0))
    plt.text(0.02, 0.98, f'注意：部分损失值超出显示范围\n最大损失: {max_loss:.2f}', 
             transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.plot(range(1, len(train_loss) + 1), train_loss, label=train_label, color='blue', linewidth=2)
plt.plot(range(1, len(test_loss) + 1), test_loss, label=test_label, color='red', linewidth=2)
plt.legend(fontsize=12)
plt.xlabel(xlabel_1, fontsize=12)
plt.ylabel(ylabel_1, fontsize=12)
plt.title(title_1, fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 保存图片
save_path = os.path.join(script_dir, '训练好的模型', 'training_results.png')
plt.tight_layout()  # 调整布局
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 保存高质量图片

plt.show()

print("训练结果图生成完成！")
print(f"图片保存位置: {save_path}")
print("="*50)
print("训练完成！")
