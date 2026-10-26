"""
数据增强模块
为CycleGAN微调提供数据增强功能
"""

import torch
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image


class RandomColorJitter:
    """随机颜色抖动"""
    
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, prob=0.3):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, img):
        if random.random() < self.prob:
            return self.color_jitter(img)
        return img


class RandomGaussianNoise:
    """随机高斯噪声"""
    
    def __init__(self, mean=0, std=0.01, prob=0.1):
        self.mean = mean
        self.std = std
        self.prob = prob
    
    def __call__(self, img):
        if random.random() < self.prob:
            img_tensor = transforms.ToTensor()(img)
            noise = torch.randn_like(img_tensor) * self.std + self.mean
            img_tensor = img_tensor + noise
            img_tensor = torch.clamp(img_tensor, 0, 1)
            return transforms.ToPILImage()(img_tensor)
        return img


def get_finetune_transforms(load_size=286, crop_size=256, flip_prob=0.5, 
                           color_jitter_prob=0.3, use_augmentation=True):
    """
    获取微调数据增强变换
    
    Args:
        load_size: 加载尺寸
        crop_size: 裁剪尺寸
        flip_prob: 翻转概率
        color_jitter_prob: 颜色抖动概率
        use_augmentation: 是否使用数据增强
    
    Returns:
        transforms.Compose: 数据变换组合
    """
    transform_list = []
    
    # 调整大小
    transform_list.append(transforms.Resize(load_size, Image.BICUBIC))
    
    if use_augmentation:
        # 随机裁剪
        transform_list.append(transforms.RandomCrop(crop_size))
        
        # 随机水平翻转
        if flip_prob > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))
        
        # 颜色抖动
        if color_jitter_prob > 0:
            transform_list.append(RandomColorJitter(prob=color_jitter_prob))
        
        # 高斯噪声
        transform_list.append(RandomGaussianNoise(prob=0.1))
    else:
        # 中心裁剪（无增强时）
        transform_list.append(transforms.CenterCrop(crop_size))
    
    # 转换为Tensor并归一化
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    return transforms.Compose(transform_list)


def get_test_transforms(load_size=512, crop_size=512):
    """
    获取测试数据变换
    
    Args:
        load_size: 加载尺寸
        crop_size: 裁剪尺寸
    
    Returns:
        transforms.Compose: 测试数据变换
    """
    return transforms.Compose([
        transforms.Resize(load_size, Image.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def apply_augmentation_to_batch(batch, augmentation_type='color_jitter'):
    """
    对批次数据应用增强
    
    Args:
        batch: 输入批次数据
        augmentation_type: 增强类型 ('color_jitter', 'gaussian_noise', 'flip')
    
    Returns:
        增强后的批次数据
    """
    if augmentation_type == 'color_jitter':
        color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
        return color_jitter(batch)
    
    elif augmentation_type == 'gaussian_noise':
        noise = torch.randn_like(batch) * 0.01
        return torch.clamp(batch + noise, 0, 1)
    
    elif augmentation_type == 'flip':
        if random.random() > 0.5:
            return torch.flip(batch, [3])  # 水平翻转
        return batch
    
    return batch


# 测试函数
if __name__ == '__main__':
    # 测试数据增强
    transform = get_finetune_transforms()
    print("数据增强变换创建成功")
    print(f"包含 {len(transform.transforms)} 个变换步骤")
