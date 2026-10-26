"""
TensorBoard日志记录模块
为CycleGAN微调提供可视化监控功能
"""

import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from PIL import Image
import torchvision.utils as vutils


class TensorboardLogger:
    """TensorBoard日志记录器"""
    
    def __init__(self, log_dir=None, experiment_name=None):
        """
        初始化TensorBoard记录器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        if log_dir is None:
            log_dir = './logs'
        
        if experiment_name:
            log_dir = os.path.join(log_dir, experiment_name)
        
        # 添加时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(log_dir, timestamp)
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        print(f"TensorBoard日志目录: {self.log_dir}")
        print(f"启动TensorBoard: tensorboard --logdir={log_dir}")
    
    def log_scalars(self, scalars_dict, global_step):
        """
        记录标量数据
        
        Args:
            scalars_dict: 标量字典 {name: value}
            global_step: 全局步数
        """
        for name, value in scalars_dict.items():
            if value is not None:
                self.writer.add_scalar(name, value, global_step)
    
    def log_images(self, images_dict, global_step, nrow=4):
        """
        记录图像数据
        
        Args:
            images_dict: 图像字典 {name: tensor}
            global_step: 全局步数
            nrow: 每行显示的图像数量
        """
        for name, images in images_dict.items():
            if images is not None and len(images) > 0:
                # 确保图像在[0,1]范围内
                images = (images + 1) / 2.0  # 从[-1,1]转换到[0,1]
                grid = vutils.make_grid(images, nrow=nrow, normalize=False)
                self.writer.add_image(name, grid, global_step)
    
    def log_histograms(self, hist_dict, global_step):
        """
        记录直方图数据
        
        Args:
            hist_dict: 直方图字典 {name: tensor}
            global_step: 全局步数
        """
        for name, values in hist_dict.items():
            if values is not None:
                self.writer.add_histogram(name, values, global_step)
    
    def log_model_graph(self, model, input_tensor):
        """
        记录模型计算图
        
        Args:
            model: 模型实例
            input_tensor: 输入张量
        """
        self.writer.add_graph(model, input_tensor)
    
    def log_learning_rates(self, lr_dict, global_step):
        """
        记录学习率
        
        Args:
            lr_dict: 学习率字典 {optimizer_name: lr}
            global_step: 全局步数
        """
        for name, lr in lr_dict.items():
            if lr is not None:
                self.writer.add_scalar(f'LearningRate/{name}', lr, global_step)
    
    def log_gan_losses(self, losses_dict, global_step, prefix=''):
        """
        记录GAN相关损失
        
        Args:
            losses_dict: 损失字典
            global_step: 全局步数
            prefix: 前缀
        """
        gan_metrics = {}
        
        for key, value in losses_dict.items():
            if value is not None:
                if 'G' in key or 'D' in key:
                    gan_metrics[f'{prefix}{key}'] = value
        
        self.log_scalars(gan_metrics, global_step)
    
    def log_cycle_consistency(self, losses_dict, global_step, prefix=''):
        """
        记录循环一致性损失
        
        Args:
            losses_dict: 损失字典
            global_step: 全局步数
            prefix: 前缀
        """
        cycle_metrics = {}
        
        for key, value in losses_dict.items():
            if value is not None:
                if 'cycle' in key.lower() or 'identity' in key.lower():
                    cycle_metrics[f'{prefix}{key}'] = value
        
        self.log_scalars(cycle_metrics, global_step)
    
    def log_gradient_norms(self, model, global_step, prefix=''):
        """
        记录梯度范数
        
        Args:
            model: 模型实例
            global_step: 全局步数
            prefix: 前缀
        """
        grad_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[f'{prefix}GradNorm/{name}'] = grad_norm
        
        self.log_scalars(grad_norms, global_step)
    
    def log_parameter_stats(self, model, global_step, prefix=''):
        """
        记录参数统计信息
        
        Args:
            model: 模型实例
            global_step: 全局步数
            prefix: 前缀
        """
        param_stats = {}
        
        for name, param in model.named_parameters():
            if param is not None:
                param_stats[f'{prefix}ParamMean/{name}'] = param.data.mean().item()
                param_stats[f'{prefix}ParamStd/{name}'] = param.data.std().item()
                param_stats[f'{prefix}ParamAbsMax/{name}'] = param.data.abs().max().item()
        
        self.log_scalars(param_stats, global_step)
    
    def close(self):
        """关闭TensorBoard写入器"""
        self.writer.close()
    
    def __del__(self):
        """析构函数，确保资源释放"""
        self.close()


def create_tensorboard_logger(config):
    """
    创建TensorBoard记录器
    
    Args:
        config: 配置对象
    
    Returns:
        TensorboardLogger实例
    """
    if config.use_tensorboard:
        return TensorboardLogger(
            log_dir=os.path.join(config.checkpoints_dir, 'tensorboard'),
            experiment_name=config.name
        )
    return None


# 工具函数
def prepare_images_for_logging(visuals, max_images=8):
    """
    准备用于日志记录的图像
    
    Args:
        visuals: 可视化字典
        max_images: 最大图像数量
    
    Returns:
        处理后的图像字典
    """
    result = {}
    
    for label, image in visuals.items():
        if image is not None:
            # 限制图像数量
            if len(image) > max_images:
                image = image[:max_images]
            result[label] = image
    
    return result


def log_training_progress(logger, losses, visuals, learning_rates, global_step, epoch):
    """
    记录训练进度
    
    Args:
        logger: TensorBoard记录器
        losses: 损失字典
        visuals: 可视化图像
        learning_rates: 学习率字典
        global_step: 全局步数
        epoch: 当前轮数
    """
    if logger is None:
        return
    
    # 记录标量损失
    logger.log_scalars(losses, global_step)
    
    # 记录GAN相关损失
    logger.log_gan_losses(losses, global_step)
    
    # 记录循环一致性损失
    logger.log_cycle_consistency(losses, global_step)
    
    # 记录学习率
    logger.log_learning_rates(learning_rates, global_step)
    
    # 记录图像（每10个epoch记录一次）
    if epoch % 10 == 0 and visuals is not None:
        prepared_images = prepare_images_for_logging(visuals)
        logger.log_images(prepared_images, global_step)


if __name__ == '__main__':
    # 测试TensorBoard记录器
    logger = TensorboardLogger(experiment_name='test')
    print("TensorBoard记录器创建成功")
    logger.close()
