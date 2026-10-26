"""
微调模型模块
扩展CycleGAN模型以支持微调功能
"""

import os
import torch
import torch.nn as nn
from models.cycle_gan_model import CycleGANModel
from models.networks import init_net
from util.util import mkdirs


class FinetuneCycleGANModel(CycleGANModel):
    """支持微调的CycleGAN模型"""
    
    def __init__(self, gpu_ids='0', isTrain=True, checkpoints_dir='./checkpoints', 
                 name='experiment_name', continue_train=False, pretrained_path=None):
        """
        初始化微调模型
        
        Args:
            pretrained_path: 预训练模型路径
        """
        super().__init__(gpu_ids, isTrain, checkpoints_dir, name, continue_train)
        self.pretrained_path = pretrained_path
        
        if self.isTrain and pretrained_path:
            print(f"使用预训练模型: {pretrained_path}")
    
    def setup(self, opt=None):
        """设置模型，支持从预训练模型加载"""
        super().setup()
        
        # 从预训练模型加载权重
        if self.isTrain and self.pretrained_path and os.path.exists(self.pretrained_path):
            self.load_pretrained_weights(self.pretrained_path)
    
    def load_pretrained_weights(self, pretrained_path):
        """加载预训练权重"""
        try:
            if os.path.exists(pretrained_path):
                print(f"加载预训练权重: {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location=self.device)
                
                # 加载生成器权重
                if 'netG_A' in state_dict:
                    self.netG_A.load_state_dict(state_dict['netG_A'])
                if 'netG_B' in state_dict:
                    self.netG_B.load_state_dict(state_dict['netG_B'])
                
                # 加载判别器权重（可选）
                if hasattr(self, 'netD_A') and 'netD_A' in state_dict:
                    self.netD_A.load_state_dict(state_dict['netD_A'])
                if hasattr(self, 'netD_B') and 'netD_B' in state_dict:
                    self.netD_B.load_state_dict(state_dict['netD_B'])
                
                print("预训练权重加载成功")
            else:
                print(f"预训练模型文件不存在: {pretrained_path}")
                
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
    
    def configure_optimizers(self, lr=0.0002, beta1=0.5, beta2=0.999):
        """配置优化器，支持微调学习率"""
        super().configure_optimizers(lr, beta1, beta2)
        
        # 微调时可以设置不同的学习率
        if hasattr(self, 'optimizer_G'):
            print(f"生成器学习率: {lr}")
        if hasattr(self, 'optimizer_D'):
            print(f"判别器学习率: {lr}")
    
    def set_input(self, input):
        """设置输入，支持数据增强"""
        super().set_input(input)
        
        # 在微调时可以添加额外的数据增强
        if self.isTrain and hasattr(self, 'real_A') and hasattr(self, 'real_B'):
            self.apply_in_batch_augmentation()
    
    def apply_in_batch_augmentation(self):
        """应用批次内数据增强"""
        # 可以在这里添加实时数据增强
        # 例如：颜色抖动、随机噪声等
        pass
    
    def optimize_parameters(self):
        """优化参数，支持梯度累积等微调技巧"""
        # 前向传播
        self.forward()
        
        # 训练生成器
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        # 训练判别器
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
    
    def get_current_learning_rates(self):
        """获取当前学习率"""
        lr_dict = {}
        
        if hasattr(self, 'optimizer_G'):
            lr_dict['Generator'] = self.optimizer_G.param_groups[0]['lr']
        if hasattr(self, 'optimizer_D'):
            lr_dict['Discriminator'] = self.optimizer_D.param_groups[0]['lr']
        
        return lr_dict
    
    def get_model_size(self):
        """获取模型大小信息"""
        size_info = {}
        
        def get_param_count(net):
            return sum(p.numel() for p in net.parameters() if p.requires_grad)
        
        if hasattr(self, 'netG_A'):
            size_info['Generator_A'] = get_param_count(self.netG_A)
        if hasattr(self, 'netG_B'):
            size_info['Generator_B'] = get_param_count(self.netG_B)
        if hasattr(self, 'netD_A'):
            size_info['Discriminator_A'] = get_param_count(self.netD_A)
        if hasattr(self, 'netD_B'):
            size_info['Discriminator_B'] = get_param_count(self.netD_B)
        
        return size_info
    
    def freeze_layers(self, layer_names=None):
        """冻结指定层"""
        if layer_names is None:
            layer_names = []
        
        for name, param in self.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False
                    print(f"冻结层: {name}")
    
    def unfreeze_all_layers(self):
        """解冻所有层"""
        for name, param in self.named_parameters():
            param.requires_grad = True
        print("所有层已解冻")
    
    def set_different_learning_rates(self, lr_generator, lr_discriminator):
        """为生成器和判别器设置不同的学习率"""
        if hasattr(self, 'optimizer_G'):
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = lr_generator
        
        if hasattr(self, 'optimizer_D'):
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr_discriminator
        
        print(f"生成器学习率: {lr_generator}, 判别器学习率: {lr_discriminator}")


def create_finetune_model(gpu_ids='0', isTrain=True, checkpoints_dir='./checkpoints', 
                         name='experiment_name', continue_train=False, pretrained_path=None):
    """
    创建微调模型
    
    Args:
        pretrained_path: 预训练模型路径
    
    Returns:
        FinetuneCycleGANModel实例
    """
    model = FinetuneCycleGANModel(
        gpu_ids=gpu_ids,
        isTrain=isTrain,
        checkpoints_dir=checkpoints_dir,
        name=name,
        continue_train=continue_train,
        pretrained_path=pretrained_path
    )
    return model


# 工具函数
def load_model_for_finetuning(model_path, device='cuda'):
    """加载模型用于微调"""
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            return checkpoint
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None
    else:
        print(f"模型文件不存在: {model_path}")
        return None


def save_finetune_checkpoint(model, epoch, iteration, save_path):
    """保存微调检查点"""
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'netG_A_state_dict': model.netG_A.state_dict(),
        'netG_B_state_dict': model.netG_B.state_dict(),
        'netD_A_state_dict': model.netD_A.state_dict() if hasattr(model, 'netD_A') else None,
        'netD_B_state_dict': model.netD_B.state_dict() if hasattr(model, 'netD_B') else None,
        'optimizer_G_state_dict': model.optimizer_G.state_dict() if hasattr(model, 'optimizer_G') else None,
        'optimizer_D_state_dict': model.optimizer_D.state_dict() if hasattr(model, 'optimizer_D') else None,
    }
    
    torch.save(checkpoint, save_path)
    print(f"检查点已保存: {save_path}")


if __name__ == '__main__':
    # 测试微调模型
    model = create_finetune_model(isTrain=True, pretrained_path='./checkpoints/first/latest_net_G.pth')
    print("微调模型创建成功")
    
    if model.isTrain:
        model.setup()
        size_info = model.get_model_size()
        print(f"模型参数数量: {size_info}")
