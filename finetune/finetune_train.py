"""
微调训练脚本
基于预训练模型进行CycleGAN微调
"""

import os
import time
import argparse
import torch
from tqdm import tqdm
import sys
import os
from torch.cuda import amp

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import create_dataset
from util.util import mkdirs, print_current_losses
from finetune.finetune_config import FinetuneConfig
from finetune.finetune_model import create_finetune_model
from finetune.tensorboard_logger import create_tensorboard_logger, log_training_progress


def main(config):
    """
    微调主函数
    
    Args:
        config: 微调配置
    """
    # 创建检查点目录
    expr_dir = os.path.join(config.checkpoints_dir, config.name)
    mkdirs(expr_dir)
    
    # 创建TensorBoard记录器
    logger = create_tensorboard_logger(config)
    
    # 创建数据集
    print(f"加载数据集: {config.dataroot}")
    dataset = create_dataset(
        config.dataroot, 
        batch_size=config.batch_size, 
        phase='train', 
        load_size=config.load_size, 
        crop_size=config.crop_size, 
        serial_batches=False, 
        input_nc=config.input_nc, 
        output_nc=config.output_nc,
        num_workers=0  # 根据25核心配置优化数据加载
    )
    dataset_size = len(dataset)
    print(f"数据集大小: {dataset_size}")
    
    # 创建微调模型
    model = create_finetune_model(
        gpu_ids=config.gpu_ids,
        isTrain=True,
        checkpoints_dir=config.checkpoints_dir,
        name=config.name,
        continue_train=False,
        pretrained_path=config.pretrained_path
    )
    
    # 设置模型
    model.setup()
    
    # 初始化混合精度训练
    scaler = None
    if torch.cuda.is_available():
        try:
            scaler = torch.amp.GradScaler('cuda')
        except:
            scaler = amp.GradScaler()
    
    # 打印模型信息
    model_size = model.get_model_size()
    print("模型参数数量:")
    for name, count in model_size.items():
        print(f"  {name}: {count:,}")
    
    total_iters = 0
    epoch_count = 1
    
    # 训练循环
    for epoch in range(epoch_count, config.n_epochs + config.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        
        # 更新学习率
        model.update_learning_rate()
        
        # 获取当前学习率
        learning_rates = model.get_current_learning_rates()
        
        # 进度条
        pbar = tqdm(dataset, desc=f'Epoch {epoch}/{config.n_epochs + config.n_epochs_decay}')
        
        for i, data in enumerate(pbar):
            iter_start_time = time.time()
            
            # 计算数据加载时间
            if total_iters % config.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            total_iters += config.batch_size
            epoch_iter += config.batch_size
            
            # 设置输入并优化参数
            model.set_input(data)
            
            # 使用混合精度训练
            if scaler is not None:
                try:
                    # 前向传播
                    with torch.amp.autocast('cuda'):
                        model.forward()
                    
                    # 训练生成器
                    model.set_requires_grad([model.netD_A, model.netD_B], False)
                    model.optimizer_G.zero_grad()
                    with torch.amp.autocast('cuda'):
                        model.backward_G()
                    scaler.scale(model.loss_G).backward()
                    scaler.step(model.optimizer_G)
                    scaler.update()
                    
                    # 训练判别器
                    model.set_requires_grad([model.netD_A, model.netD_B], True)
                    model.optimizer_D.zero_grad()
                    model.backward_D_A()
                    model.backward_D_B()
                    model.optimizer_D.step()
                except:
                    # 回退到标准训练
                    model.optimize_parameters()
            else:
                model.optimize_parameters()
            
            # 实时更新进度条描述（损失信息）
            if total_iters % 10 == 0:  # 每10个迭代更新一次
                losses = model.get_current_losses()
                loss_desc = ', '.join([f'{k}:{v:.3f}' for k, v in losses.items() if v is not None])
                pbar.set_postfix_str(loss_desc)
            
            # 记录损失和可视化（保持原有打印频率）
            if total_iters % config.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / config.batch_size
                
                # 记录到TensorBoard
                if logger:
                    visuals = model.get_current_visuals()
                    log_training_progress(
                        logger, losses, visuals, learning_rates, total_iters, epoch
                    )
            
            iter_data_time = time.time()
        
        # 保存模型
        if epoch % config.save_epoch_freq == 0:
            print(f'保存模型在 epoch {epoch}, iterations {total_iters}')
            model.save_networks('latest')
            model.save_networks(epoch)
        
        # 输出epoch信息
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch}/{config.n_epochs + config.n_epochs_decay} 完成, 耗时: {epoch_time:.1f}秒')
        
        # 记录epoch信息到TensorBoard
        if logger:
            logger.log_scalars({'Epoch/Time': epoch_time}, epoch)
    
    # 关闭TensorBoard记录器
    if logger:
        logger.close()
    
    print("微调训练完成！")


if __name__ == '__main__':
    # 从命令行参数获取配置
    config = FinetuneConfig.from_args()
    
    # 打印配置信息
    print("=" * 50)
    print("CycleGAN微调配置")
    print("=" * 50)
    print(f"数据集目录: {config.dataroot}")
    print(f"预训练模型: {config.pretrained_path}")
    print(f"检查点目录: {config.checkpoints_dir}")
    print(f"实验名称: {config.name}")
    print(f"训练轮数: {config.n_epochs} + {config.n_epochs_decay} decay")
    print(f"学习率: {config.lr}")
    print(f"批大小: {config.batch_size}")
    print(f"使用TensorBoard: {config.use_tensorboard}")
    print(f"使用数据增强: {config.use_data_augmentation}")
    print("=" * 50)
    
    # 检查预训练模型是否存在
    if not os.path.exists(config.pretrained_path):
        print(f"警告: 预训练模型文件不存在: {config.pretrained_path}")
        print("将继续从头开始训练...")
    
    # 检查数据集目录是否存在
    if not os.path.exists(config.dataroot):
        print(f"错误: 数据集目录不存在: {config.dataroot}")
        print("请创建数据集目录或使用 --dataroot 参数指定正确的路径")
        print("数据集格式应与原始CycleGAN相同: trainA/ 和 trainB/ 文件夹")
        exit(1)
    
    # 开始训练
    try:
        main(config)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


# 命令行使用示例
"""
python CycleGAN/finetune/finetune_train.py \
    --dataroot ./datasets/chinesepainting_finetune \
    --pretrained_path ./checkpoints/first/latest_net_G.pth \
    --n_epochs 100 \
    --n_epochs_decay 100 \
    --lr 0.0002 \
    --batch_size 1 \
    --name chinesepainting_finetune \
    --gpu_ids 0
"""
