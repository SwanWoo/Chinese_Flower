# Chinese Flower Painting CycleGAN

基于 CycleGAN 的中式花鸟画风格转换项目。本项目使用 CycleGAN 模型实现自然风景图像到中国传统花鸟画风格的转换。

## 项目特性

- ✅ 自然风景 ↔ 中国画风格双向转换
- ✅ 支持 CycleGAN 和 Pix2Pix 模型
- ✅ 预训练模型微调功能
- ✅ TensorBoard 可视化训练过程
- ✅ GPU 加速训练
- ✅ 支持多 GPU 并行训练
- ✅ 数据增强功能
- ✅ 高性能优化方案（GPU利用率提升50-80%）

## 目录结构

```
Chinese_Flower_Painting_CycleGAN/
├── data/                     # 数据加载模块
│   ├── __init__.py          # 数据集工厂
│   ├── base_dataset.py      # 基础数据集类
│   ├── unaligned_dataset.py # 无配对数据集
│   └── image_folder.py      # 图像文件夹处理
├── models/                  # 模型定义
│   ├── __init__.py          # 模型工厂
│   ├── base_model.py        # 基础模型类
│   ├── cycle_gan_model.py   # CycleGAN 模型
│   └── networks.py          # 网络架构
├── finetune/               # 微调模块
│   ├── finetune_train.py   # 微调训练脚本
│   ├── fast_finetune.py    # 快速微调（带性能优化）
│   └── README.md           # 微调说明
├── util/                   # 工具函数
│   └── util.py             # 工具函数
├── train.py                # 主训练脚本
├── test.py                 # 测试脚本
└── README.md              # 本文档
```

## 环境配置

### 硬件要求
- GPU: NVIDIA GPU (建议 8GB+ 显存)
- 内存: 16GB+ (推荐 32GB)
- 存储: SSD (推荐，减少 IO 瓶颈)

### 软件依赖

```bash
# 基础依赖
pip install torch torchvision
pip install Pillow numpy scikit-image
pip install tqdm matplotlib

# 可选依赖（用于可视化）
pip install tensorboard visdom

# 性能监控
pip install gpustat nvidia-ml-py3
```

### PyTorch 版本要求
- PyTorch 1.7+ (推荐 1.13+ 或 2.0+)
- CUDA 11.0+ (如使用 GPU)

## 数据准备

### 数据集结构

数据集需要按以下结构组织：

```
datasets/
├── chinesepainting/        # 训练数据集
│   ├── trainA/            # 域A：自然风景
│   └── trainB/            # 域B：中国画
└── nature_image/          # 测试数据集
    └── testA/             # 待转换的自然风景图像
```

### 图像要求
- 格式: JPG/PNG
- 尺寸: 建议 256×256 或 512×512
- 颜色空间: RGB

### 准备示例数据集

1. **收集自然风景图像** 放入 `datasets/chinesepainting/trainA/`
2. **收集中国画图像** 放入 `datasets/chinesepainting/trainB/`
3. **准备测试图像** 放入 `datasets/nature_image/testA/`

## 快速开始

### 1. 训练 CycleGAN 模型

```bash
# 基础训练（默认参数）
python train.py \
    --dataroot ./datasets/chinesepainting \
    --name chinesepainting_cyclegan \
    --gpu_id 0

# 优化训练（推荐，GPU利用率更高）
python train.py \
    --dataroot ./datasets/chinesepainting \
    --name optimized_cyclegan \
    --gpu_id 0
```

### 2. 测试风格转换

```bash
# 使用训练好的模型进行风格转换
python test.py \
    --dataroot ./datasets/nature_image \
    --name chinesepainting_cyclegan \
    --result_dir ./results

# 指定模型检查点
python test.py \
    --dataroot ./datasets/nature_image \
    --name chinesepainting_cyclegan \
    --result_dir ./results/epoch_100
```

## 训练参数详解

### 基本参数
```bash
python train.py \
    --dataroot ./datasets/chinesepainting \  # 数据集路径
    --n_epochs 200 \                         # 训练轮数
    --n_epochs_decay 200 \                   # 学习率衰减轮数
    --gpu_id 0 \                             # GPU ID (-1 表示 CPU)
    --checkpoints_dir ./checkpoints \        # 模型保存目录
    --name experiment_name                   # 实验名称
```

### 数据参数
```bash
python train.py \
    --load_size 286 \    # 图像加载尺寸
    --crop_size 256 \    # 随机裁剪尺寸
    --serial_batches \   # 禁用随机打乱
    --no_flip \          # 禁用水平翻转
```

## 微调训练

### 使用预训练模型微调

```bash
python finetune/finetune_train.py \
    --dataroot ./datasets/chinesepainting_finetune \
    --pretrained_path ./checkpoints/chinesepainting_cyclegan/latest_net_G.pth \
    --n_epochs 100 \
    --n_epochs_decay 100 \
    --lr 0.0001 \
    --batch_size 2 \
    --name chinesepainting_finetune \
    --gpu_ids 0
```

### 快速微调（带性能优化）

```bash
python finetune/fast_finetune.py \
    --dataroot ./datasets/chinesepainting_finetune \
    --pretrained_path ./checkpoints/chinesepainting_cyclegan/latest_net_G.pth \
    --n_epochs 50 \
    --batch_size 4 \
    --num_workers 8 \
    --use_cudnn_benchmark \
    --use_mixed_precision \
    --name fast_finetune
```

## 性能优化指南

### 问题：GPU 利用率低（30-50%）

**主要原因**：同步图像 IO 阻塞，GPU 等待数据加载。

### 优化方案（详细见 [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md)）

#### 1. 优化 DataLoader 配置（最有效）
修改 `data/__init__.py` 中的默认参数：

```python
# 修改前：batch_size=1, num_workers=4
def create_dataset(..., batch_size=1, num_workers=4):
    ...

# 修改后：batch_size=8, num_workers=8
def create_dataset(..., batch_size=8, num_workers=8):
    self.dataloader = torch.utils.data.DataLoader(
        ...
        prefetch_factor=2  # 新增数据预取
    )
```

**参数建议**：
- `batch_size`: 4-16（根据显存调整，从 8 开始）
- `num_workers`: 8-16（根据 CPU 核心数调整，建议 8）
- `prefetch_factor`: 2-4（建议 2）

#### 2. 异步模型保存
修改 `models/base_model.py` 避免保存时的 IO 阻塞：

```python
import threading
from concurrent.futures import ThreadPoolExecutor

class BaseModel(ABC):
    def __init__(self, ...):
        self.save_executor = ThreadPoolExecutor(max_workers=1)
        self.save_thread = None

    def save_networks(self, epoch):
        """异步保存模型"""
        if self.save_thread is not None:
            self.save_thread.join()

        def _save_task():
            # 保存逻辑...

        self.save_thread = self.save_executor.submit(_save_task)
```

#### 3. 数据缓存（内存充足时）
创建 `data/cached_dataset.py`：

```python
class CachedUnalignedDataset(UnalignedDataset):
    def __init__(self, ..., cache_size=1000):
        super().__init__(...)
        self.cache = {}  # 内存缓存
        self.cache_size = cache_size

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]  # 缓存命中
        # 缓存未命中，从磁盘加载...
```

### 优化效果对比

| 优化方案 | GPU 利用率提升 | 训练速度提升 | 实施难度 |
|---------|---------------|-------------|---------|
| DataLoader 配置优化 | +20-40% | 1.5-2x | 低 |
| 异步模型保存 | +5-10% | 1.05-1.1x | 中 |
| 数据缓存 | +30-50% | 2-3x | 中 |
| **综合优化** | **+50-80%** | **2-4x** | - |

### GPU 利用率监控

```bash
# 实时监控 GPU 利用率
nvidia-smi -l 1

# 使用 gpustat（更详细）
pip install gpustat
gpustat -i 1

# 监控显存使用
watch -n 1 nvidia-smi
```

## 多 GPU 训练

### 使用多个 GPU

```bash
# 使用 GPU 0 和 1
python train.py --gpu_id 0,1

# 使用所有可用 GPU
python train.py --gpu_id all
```

### 注意事项
1. `batch_size` 是每个 GPU 的批大小
2. 总批大小 = `batch_size` × GPU 数量
3. 确保 GPU 间有足够带宽（NVLink 或 PCIe 4.0）

## 模型架构

### CycleGAN 架构
- **生成器**: ResNet-9blocks 或 U-Net
- **判别器**: PatchGAN (70×70)
- **损失函数**:
  - GAN 损失 (LSGAN)
  - 循环一致性损失
  - 身份损失 (可选)

### 模型配置
```python
# 生成器类型：resnet_9blocks, unet_256
--netG resnet_9blocks

# 判别器类型：basic, n_layers, pixel
--netD basic

# 残差块数量
--n_blocks 9

# 判别器层数
--n_layers_D 3
```

## 可视化监控

### TensorBoard 监控

```bash
# 启动 TensorBoard
tensorboard --logdir=./checkpoints

# 访问 http://localhost:6006
```

### 监控内容
- **损失曲线**: GAN 损失、循环一致性损失
- **生成图像**: 每 10 个 epoch 保存一次
- **学习率**: 学习率变化曲线
- **梯度**: 梯度范数统计

## 常见问题

### 1. 显存不足
**解决方案**：
- 减小 `batch_size`（如从 8 减到 4）
- 减小 `crop_size`（如从 256 减到 128）
- 使用梯度累积

### 2. 训练速度慢
**解决方案**：
- 使用 SSD 替代 HDD
- 增加 `num_workers`（建议 8-16）
- 启用 `prefetch_factor`（建议 2）
- 使用数据缓存

### 3. 模型不收敛
**解决方案**：
- 检查数据集质量
- 调整学习率（尝试 0.0002、0.0001）
- 增加 `identity_loss` 权重
- 使用预训练模型微调

### 4. 图像质量差
**解决方案**：
- 增加训练轮数（200+ epochs）
- 使用更大的数据集
- 尝试不同的生成器架构
- 调整 `lambda_cycle` 和 `lambda_identity`

## 进阶用法

### 自定义数据集
创建自定义数据集类：

```python
from data.unaligned_dataset import UnalignedDataset

class CustomDataset(UnalignedDataset):
    def __init__(self, ...):
        super().__init__(...)
        # 自定义初始化

    def __getitem__(self, index):
        # 自定义数据加载逻辑
        return data
```

### 自定义损失函数
修改 `models/cycle_gan_model.py`：

```python
def backward_G(self):
    # 计算 GAN 损失
    self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
    self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

    # 自定义损失
    self.loss_custom = self.compute_custom_loss()

    # 总损失
    self.loss_G = (self.loss_G_A + self.loss_G_B) * 1.0 + \
                  self.loss_cycle_A * self.lambda_A + \
                  self.loss_cycle_B * self.lambda_B + \
                  self.loss_custom * self.lambda_custom
```

### 模型导出
```python
# 导出生成器
torch.save(model.netG_A.state_dict(), 'generator_A.pth')

# 导出 ONNX 格式（可选）
torch.onnx.export(model.netG_A, dummy_input, 'generator_A.onnx')
```

## 示例效果

### 转换示例
```
自然风景 → 中国画风格
┌─────────────┐    ┌─────────────┐
│  山水照片   │ →  │  水墨山水   │
└─────────────┘    └─────────────┘
```

### 训练日志
```
Epoch: 100/200 | Iter: 1000/10000
Loss_G: 1.234 | Loss_D: 0.876
Cycle_A: 0.456 | Cycle_B: 0.432
Time: 0.123s/iter | GPU: 85%
```

## 注意事项

### 训练建议
1. **数据集平衡**: 两个域的图像数量尽量相近
2. **图像质量**: 使用高质量、清晰的图像
3. **训练时间**: CycleGAN 需要较长时间训练（200+ epochs）
4. **监控**: 使用 TensorBoard 实时监控训练过程

### 性能建议
1. **硬件**: 使用 SSD 存储和足够的内存
2. **IO 优化**: 实施数据缓存和预取
3. **GPU**: 确保 CUDA 和 cuDNN 版本匹配
4. **内存**: 监控显存使用，避免溢出

### 模型保存
1. **检查点**: 每 epoch 保存模型
2. **最佳模型**: 根据验证损失保存最佳模型
3. **备份**: 定期备份重要检查点

## 故障排除

### 常见错误

1. **CUDA out of memory**
   ```bash
   # 解决方案
   1. 减小 batch_size
   2. 减小 crop_size
   3. 使用梯度累积
   ```

2. **找不到数据集**
   ```bash
   # 检查数据集路径
   ls ./datasets/chinesepainting/
   # 应有 trainA/ 和 trainB/ 目录
   ```

3. **模型加载失败**
   ```bash
   # 检查模型路径
   ls ./checkpoints/chinesepainting_cyclegan/
   # 应有 latest_net_G.pth 等文件
   ```

### 调试建议
1. 添加调试日志
2. 使用小数据集测试
3. 逐步增加模型复杂度
4. 监控 GPU 利用率

## 贡献与支持

### 问题反馈
如遇到问题，请：
1. 检查日志文件
2. 确认环境配置
3. 提供重现步骤
4. 附上相关截图

## 许可证

本项目基于 MIT 许可证开源。