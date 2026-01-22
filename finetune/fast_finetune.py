"""
快速微调脚本 - 针对CycleGAN的加速微调方案
包含多种性能优化技术
"""

import os
import time
import torch
import torch.nn as nn
from torch.cuda import amp
from tqdm import tqdm
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import create_dataset
from util.util import mkdirs, print_current_losses
from finetune.finetune_config import FinetuneConfig
from finetune.finetune_model import create_finetune_model
from finetune.tensorboard_logger import create_tensorboard_logger, log_training_progress


class FastFinetuneTrainer:
    """快速微调训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f'cuda:{config.gpu_ids}' if config.gpu_ids != '-1' else 'cpu')
        
        # 设置性能优化
        self._setup_performance_optimizations()
        
        # 创建目录
        self.expr_dir = os.path.join(config.checkpoints_dir, config.name)
        mkdirs(self.expr_dir)
        
        # 创建数据集
        print(f"加载数据集: {config.dataroot}")
        self.dataset = create_dataset(
            config.dataroot, 
            batch_size=config.batch_size, 
            phase='train', 
            load_size=config.load_size, 
            crop_size=config.crop_size, 
            serial_batches=False, 
            input_nc=config.input_nc, 
            output_nc=config.output_nc,
            num_workers=config.num_workers
        )
        self.dataset_size = len(self.dataset)
        print(f"数据集大小: {self.dataset_size}")
        
        # 创建模型
        self.model = create_finetune_model(
            gpu_ids=config.gpu_ids,
            isTrain=True,
            checkpoints_dir=config.checkpoints_dir,
            name=config.name,
            continue_train=False,
            pretrained_path=config.pretrained_path
        )
        
        # 设置模型
        self.model.setup()
        
        # 应用梯度检查点
        if config.use_gradient_checkpointing:
            self._apply_gradient_checkpointing()
        
        # 创建TensorBoard记录器
        self.logger = create_tensorboard_logger(config) if config.use_tensorboard else None
        
        # 初始化混合精度训练
        self.scaler = amp.GradScaler() if config.use_amp and torch.cuda.is_available() else None
        
        # 训练状态
        self.total_iters = 0
        self.epoch_count = 1
        
    def _setup_performance_optimizations(self):
        """设置性能优化配置"""
        if self.config.use_cudnn_benchmark and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        if self.config.pin_memory and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)  # 预留10%内存给系统
        
    def _apply_gradient_checkpointing(self):
        """应用梯度检查点技术"""
        def apply_checkpointing(module):
            if hasattr(module, 'apply_gradient_checkpointing'):
                module.apply_gradient_checkpointing()
            for child in module.children():
                apply_checkpointing(child)
        
        apply_checkpointing(self.model.netG_A)
        apply_checkpointing(self.model.netG_B)
        print("已启用梯度检查点技术")
    
    def _gradient_accumulation_step(self, data, accumulation_steps):
        """梯度累积步骤"""
        losses = {}
        
        # 设置输入
        self.model.set_input(data)
        
        # 使用混合精度训练
        if self.scaler is not None:
            try:
                # 前向传播
                with amp.autocast():
                    self.model.forward()
                
                # 训练生成器
                self.model.set_requires_grad([self.model.netD_A, self.model.netD_B], False)
                self.model.optimizer_G.zero_grad()
                
                with amp.autocast():
                    self.model.backward_G()
                
                # 缩放损失并反向传播
                self.scaler.scale(self.model.loss_G / accumulation_steps).backward()
                
                # 训练判别器
                self.model.set_requires_grad([self.model.netD_A, self.model.netD_B], True)
                self.model.optimizer_D.zero_grad()
                
                self.model.backward_D_A()
                self.model.backward_D_B()
                
                # 缩放判别器损失
                self.scaler.scale(self.model.loss_D / accumulation_steps).backward()
                
            except:
                # 回退到标准训练
                self.model.optimize_parameters()
        else:
            self.model.optimize_parameters()
        
        # 收集损失
        losses.update(self.model.get_current_losses())
        return losses
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        epoch_start_time = time.time()
        epoch_iter = 0
        
        # 更新学习率
        self.model.update_learning_rate()
        learning_rates = self.model.get_current_learning_rates()
        
        # 进度条
        pbar = tqdm(self.dataset, desc=f'Epoch {epoch}/{self.config.n_epochs + self.config.n_epochs_decay}')
        
        accumulation_steps = self.config.gradient_accumulation_steps
        accumulation_counter = 0
        
        for i, data in enumerate(pbar):
            iter_start_time = time.time()
            
            # 梯度累积步骤
            losses = self._gradient_accumulation_step(data, accumulation_steps)
            accumulation_counter += 1
            
            # 更新迭代计数
            self.total_iters += self.config.batch_size
            epoch_iter += self.config.batch_size
            
            # 达到累积步数时更新参数
            if accumulation_counter % accumulation_steps == 0:
                if self.scaler is not None:
                    # 更新生成器
                    self.scaler.step(self.model.optimizer_G)
                    # 更新判别器
                    self.scaler.step(self.model.optimizer_D)
                    # 更新缩放器
                    self.scaler.update()
                else:
                    self.model.optimizer_G.step()
                    self.model.optimizer_D.step()
                
                # 重置梯度
                self.model.optimizer_G.zero_grad()
                self.model.optimizer_D.zero_grad()
            
            # 实时更新进度条
            if self.total_iters % 10 == 0:
                loss_desc = ', '.join([f'{k}:{v:.3f}' for k, v in losses.items() if v is not None])
                pbar.set_postfix_str(loss_desc)
            
            # 记录损失和可视化
            if self.total_iters % self.config.print_freq == 0:
                t_comp = (time.time() - iter_start_time) / self.config.batch_size
                
                # 记录到TensorBoard
                if self.logger:
                    visuals = self.model.get_current_visuals()
                    log_training_progress(
                        self.logger, losses, visuals, learning_rates, self.total_iters, epoch
                    )
        
        # 保存模型
        if epoch % self.config.save_epoch_freq == 0:
            print(f'保存模型在 epoch {epoch}, iterations {self.total_iters}')
            self.model.save_networks('latest')
            self.model.save_networks(epoch)
        
        # 输出epoch信息
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch}/{self.config.n_epochs + self.config.n_epochs_decay} 完成, 耗时: {epoch_time:.1f}秒')
        
        # 记录epoch信息到TensorBoard
        if self.logger:
            self.logger.log_scalars({'Epoch/Time': epoch_time}, epoch)
    
    def train(self):
        """执行训练"""
        try:
            for epoch in range(self.epoch_count, self.config.n_epochs + self.config.n_epochs_decay + 1):
                self.train_epoch(epoch)
            
            print("快速微调训练完成！")
            
        except KeyboardInterrupt:
            print("\n训练被用户中断")
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 关闭TensorBoard记录器
            if self.logger:
                self.logger.close()


def main():
    """主函数"""
    # 从命令行参数获取配置
    config = FinetuneConfig.from_args()
    
    # 打印配置信息
    print("=" * 50)
    print("CycleGAN快速微调配置")
    print("=" * 50)
    print(f"数据集目录: {config.dataroot}")
    print(f"预训练模型: {config.pretrained_path}")
    print(f"批大小: {config.batch_size}")
    print(f"梯度累积步数: {config.gradient_accumulation_steps}")
    print(f"数据加载工作进程: {config.num_workers}")
    print(f"混合精度训练: {config.use_amp}")
    print(f"梯度检查点: {config.use_gradient_checkpointing}")
    print(f"CuDNN基准测试: {config.use_cudnn_benchmark}")
    print(f"内存锁定: {config.pin_memory}")
    print(f"持久化工作进程: {config.persistent_workers}")
    print("=" * 50)
    
    # 检查预训练模型是否存在
    if not os.path.exists(config.pretrained_path):
        print(f"警告: 预训练模型文件不存在: {config.pretrained_path}")
        print("将继续从头开始训练...")
    
    # 检查数据集目录是否存在
    if not os.path.exists(config.dataroot):
        print(f"错误: 数据集目录不存在: {config.dataroot}")
        print("请创建数据集目录或使用 --dataroot 参数指定正确的路径")
        exit(1)
    
    # 创建并启动训练器
    trainer = FastFinetuneTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()


# 快速启动命令示例
"""
python finetune/fast_finetune.py \
    --dataroot ./dataroot/chinesepainting_finetune \
    --pretrained_path ./checkpoints/first/latest_net_G.pth \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_workers 16 \
    --n_epochs 50 \
    --n_epochs_decay 50 \
    --lr 0.0001 \
    --name chinesepainting_fast_finetune \
    --gpu_ids 0
"""
