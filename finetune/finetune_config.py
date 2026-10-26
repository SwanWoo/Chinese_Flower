"""
微调配置文件
定义CycleGAN微调的超参数和配置
"""

import argparse
from dataclasses import dataclass


@dataclass
class FinetuneConfig:
    """微调配置类"""
    
    # 数据相关配置
    dataroot: str = './datasets/chinesepainting_finetune'
    load_size: int = 286
    crop_size: int = 256
    batch_size: int = 1
    input_nc: int = 3
    output_nc: int = 3
    
    # 训练相关配置
    n_epochs: int = 100
    n_epochs_decay: int = 100
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    
    # 模型相关配置
    pretrained_path: str = './checkpoints/first/latest_net_G.pth'
    checkpoints_dir: str = './checkpoints/finetune'
    name: str = 'chinesepainting_finetune'
    
    # 设备配置
    gpu_ids: str = '0'
    
    # 日志和保存配置
    print_freq: int = 100
    save_epoch_freq: int = 1
    use_tensorboard: bool = True
    
    # 性能优化配置
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    use_gradient_checkpointing: bool = True
    use_cudnn_benchmark: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # 数据增强配置
    use_data_augmentation: bool = True
    flip_prob: float = 0.5
    color_jitter_prob: float = 0.3
    
    @classmethod
    def from_args(cls):
        """从命令行参数创建配置"""
        parser = argparse.ArgumentParser(description='CycleGAN微调配置')
        
        # 数据相关参数
        parser.add_argument('--dataroot', default=cls.dataroot, 
                          help='微调数据集目录')
        parser.add_argument('--load_size', type=int, default=cls.load_size,
                          help='图像加载尺寸')
        parser.add_argument('--crop_size', type=int, default=cls.crop_size,
                          help='图像裁剪尺寸')
        parser.add_argument('--batch_size', type=int, default=cls.batch_size,
                          help='批大小')
        
        # 训练相关参数
        parser.add_argument('--n_epochs', type=int, default=cls.n_epochs,
                          help='训练轮数')
        parser.add_argument('--n_epochs_decay', type=int, default=cls.n_epochs_decay,
                          help='学习率衰减轮数')
        parser.add_argument('--lr', type=float, default=cls.lr,
                          help='学习率')
        
        # 模型相关参数
        parser.add_argument('--pretrained_path', default=cls.pretrained_path,
                          help='预训练模型路径')
        parser.add_argument('--checkpoints_dir', default=cls.checkpoints_dir,
                          help='检查点保存目录')
        parser.add_argument('--name', default=cls.name,
                          help='实验名称')
        
        # 设备参数
        parser.add_argument('--gpu_ids', default=cls.gpu_ids,
                            help='GPU ID，-1表示使用CPU')
            
        # 日志参数
        parser.add_argument('--print_freq', type=int, default=cls.print_freq,
                            help='打印频率')
        parser.add_argument('--save_epoch_freq', type=int, default=cls.save_epoch_freq,
                            help='保存频率')
        parser.add_argument('--no_tensorboard', action='store_true',
                            help='禁用TensorBoard')
            
        # 数据增强参数
        parser.add_argument('--no_data_aug', action='store_true',
                            help='禁用数据增强')

        # 性能优化参数
        parser.add_argument('--gradient_accumulation_steps', type=int, default=cls.gradient_accumulation_steps,
                            help='梯度累积步数')
        parser.add_argument('--num_workers', type=int, default=cls.num_workers,
                            help='数据加载工作进程数')
        parser.add_argument('--no_amp', action='store_true',
                            help='禁用混合精度训练')
        parser.add_argument('--no_gradient_checkpointing', action='store_true',
                            help='禁用梯度检查点')
        parser.add_argument('--no_cudnn_benchmark', action='store_true',
                            help='禁用CuDNN基准测试')
        parser.add_argument('--no_pin_memory', action='store_true',
                            help='禁用内存锁定')
        parser.add_argument('--no_persistent_workers', action='store_true',
                            help='禁用持久化工作进程')
            
        args = parser.parse_args()
            
        return cls(
            dataroot=args.dataroot,
            load_size=args.load_size,
            crop_size=args.crop_size,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            n_epochs_decay=args.n_epochs_decay,
            lr=args.lr,
            pretrained_path=args.pretrained_path,
            checkpoints_dir=args.checkpoints_dir,
            name=args.name,
            gpu_ids=args.gpu_ids,
            print_freq=args.print_freq,
            save_epoch_freq=args.save_epoch_freq,
            use_tensorboard=not args.no_tensorboard,
            use_data_augmentation=not args.no_data_aug,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_workers=args.num_workers,
            use_amp=not args.no_amp,
            use_gradient_checkpointing=not args.no_gradient_checkpointing,
            use_cudnn_benchmark=not args.no_cudnn_benchmark,
            pin_memory=not args.no_pin_memory,
            persistent_workers=not args.no_persistent_workers
        )


def get_default_config():
    """获取默认配置"""
    return FinetuneConfig()
