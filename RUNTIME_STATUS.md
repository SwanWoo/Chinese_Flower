# 运行时状态与训练性能报告

- **生成时间**：2026-10-26 02:00 (CST)
- **项目目录**：`/root/autodl-tmp/Chinese_Flower`（CycleGAN 照片 → 中国画风格迁移）
- **当前状态**：无正在运行的训练任务，GPU 空闲
- **一句话结论**：环境可训练；优化后推荐 `--batch_size 4` + 关闭 identity loss + fused Adam，单 epoch 约 **3.6 分钟**（旧默认约 15.2 分钟，历史日志约 26.8 分钟）

---

## 1. 硬件与环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER，32,760 MiB 显存 |
| 驱动 / CUDA | 595.71.05 / 13.2（驱动报告） |
| GPU 当前状态 | 37°C，11.9W，210MHz，0% 利用率（空闲） |
| CPU | Intel Xeon Platinum 8375C @ 2.90GHz，cgroup 限制 12 vCPU |
| 内存 | 503 GiB（可用 462 GiB） |
| 系统盘数据盘 | `/root/autodl-tmp` 50 GiB，已用 9.2 GiB，**可用约 41 GiB** |
| OS | Linux 5.15.0-25-generic x86_64 |
| Python | 3.12.3（`/root/miniconda3/bin/python`，conda base 环境） |
| PyTorch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |
| cuDNN | 9.7.1 |
| numpy / Pillow | 2.2.6 / 11.2.1 |
| 其他 | tqdm 4.66.2、TensorBoard 2.19.0；训练路径不依赖 transformers/diffusers/clip |

> ⚠️ `requirements.txt` 中的版本（torch 2.14.0、Pillow 9.5.0 等）与当前环境不符，且 Pillow 9.5.0 没有 Python 3.12 的 Linux wheel，**不要执行 `pip install -r requirements.txt`**。当前已安装环境可直接训练。

---

## 2. 数据与模型产物

### 2.1 数据集

| 目录 | 文件数 | 大小 | 说明 |
|---|---:|---:|---|
| `datasets/chinesepainting/trainA` | 7,283 | 91 MB | 自然照片，JPG，320×180 |
| `datasets/chinesepainting/trainB` | 8,731 | 3.0 GB | 中国画，PNG，尺寸不一（含 RGBA） |
| `datasets/chinesepainting_finetune/trainA` | 1,638 | 70 MB | 微调用 A 域 |
| `datasets/chinesepainting_finetune/trainB` | 1,292 | 132 MB | 微调用 B 域 |
| `datasets/nature_image/testA` | 14 | 188 KB | 推理测试图 |

- 全部 18,971 张图已用 PIL 校验，0 张损坏。
- loader 会统一 `convert('RGB')`，RGBA/CMYK/灰度图均可正常读取。
- 训练时 resize 到 286×286，再随机裁剪 256×256，并做水平翻转。

### 2.2 Checkpoint

| 目录 | 大小 | 状态 |
|---|---:|---|
| `checkpoints/chinesepainting_amp_short` | 648 MB | 已完成 5 epoch，`training_state` = epoch 5 / 43,655 iters |
| `checkpoints/chinesepainting_warmstart_short` | 324 MB | 已完成 2 epoch，`training_state` = epoch 2 / 17,462 iters（最新可续训点） |
| `checkpoints/pretrained` | 87 MB | `model_init.pth` / `model_finetune.pth`，48/48 键可被 `--pretrained_g_a` 加载 |
| `checkpoints/smoke_test` | 324 MB | 2026-09-20 的旧 smoke 产物 |

- 每个 epoch checkpoint 约 108 MB（2 个生成器 + 2 个判别器）。
- `--save_epoch_freq 1` 跑 400 epoch 需约 43 GB，会超过剩余磁盘；建议 `--save_epoch_freq 5`（约 8.6 GB）。

---

## 3. 训练入口与关键参数

训练入口：`train.py`。只支持以下参数（README 中 `--gpu_ids`、`--model`、`--netG`、`--netD` 等写法**无效**）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dataroot` | `./datasets/chinesepainting` | 需包含 `trainA/`、`trainB/` |
| `--name` | `chinesepainting_cyclegan` | 实验名，输出到 `checkpoints/<name>/` |
| `--n_epochs` | 200 | 恒定学习率 epoch 数 |
| `--n_epochs_decay` | 200 | 线性衰减 epoch 数 |
| `--batch_size` | 1 | 推荐改为 4 |
| `--num_workers` | 8 | 实测 1 即可跑满，2 更稳 |
| `--load_size` / `--crop_size` | 286 / 256 | 预处理尺寸 |
| `--gpu_id` | `'0'` | `-1` 为 CPU |
| `--checkpoints_dir` | `./checkpoints` | 输出根目录 |
| `--save_epoch_freq` | 1 | 建议 5 |
| `--continue_train` | False | 从 latest checkpoint 续训 |
| `--no_amp` / `--no_compile` | False | 默认开启 AMP + torch.compile |
| `--lambda_identity` | 0.5 | 0 表示完全关闭 identity loss |
| `--identity_off_after` | 0 | N 个 epoch 后关闭 identity loss |
| `--pretrained_g_a` | `''` | 用预训练生成器热启动 |
| `--lr_g_a_scale` | 1.0 | 预训练 G_A 的学习率倍率 |
| `--freeze_g_a_epochs` | 0 | 前 N 个 epoch 冻结 G_A |

---

## 4. 瓶颈分析（实测）

1. **`batch_size=1` 是最大瓶颈**
   - GPU 利用率只有 **34%**，迭代时间主要被 kernel 启动/调度占掉，而不是算力。
   - 加到 `batch_size=4` 后利用率升到 **92%**，吞吐从 10.5 → 39.7 img/s（约 3.8 倍）。

2. **从 batch 扫描可以反推出固定每迭代开销**
   - bs1–bs4 的单批墙钟时间几乎不变：95.0 / 100.6 / 101.4 / 100.8 ms，但每批图片数从 1 变成了 4。
   - 说明这一区间由与 batch 大小基本无关的固定成分主导（CPU 调度、kernel 启动、优化器更新、GradScaler 同步），GPU 数学计算并未吃满。
   - 到 bs8 单批耗时跳到 184 ms，才进入 GPU 计算受限区；吞吐也从 bs4 的 39.7 只涨到 43.0 img/s。

3. **identity loss 约占 25–30% 计算量**
   - 开启时每步多做 2 次生成器前向 + 反向。
   - bs4 下：开启约 30 img/s，关闭约 40 img/s。

4. **优化器未使用 fused 模式**
   - 每个迭代对约 45M 参数做 Adam 更新，普通实现会产生大量小 kernel。
   - 改为 `fused=True` 后实测快 **6–14%**。

5. **数据加载不是瓶颈**
   - 解码 + H2D 每批平均 2–5 ms，而 GPU 计算约 95–140 ms。
   - `num_workers ≥ 1` 时 `data_wait ≈ 0.2 ms`；`num_workers=0` 时升到 65 ms，吞吐掉 32%。
   - 因此 README 旧版“IO 阻塞 GPU”的判断不适用于当前实现。

6. **热降频（历史问题）**
   - 2026-10-25 的历史日志中 GPU 长期处于 **~86°C / ~1500MHz**，而正常状态约 2700MHz。
   - 降频会让吞吐掉约 40%，这可能是旧运行 26.8 min/epoch 的主要原因。
   - 本次测试稳定在 **58–68°C / 2730MHz / 250–280W**，未出现降频。
   - 监控命令：
     ```bash
     nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,fan.speed --format=csv -l 5
     ```

---

## 5. 性能基准

测试方法：真实数据集 + 真实 `train.py` 模型代码，AMP + `torch.compile`，fused Adam，identity 关闭；预热后测 25 秒，GPU 指标 1 秒采样。

### 5.1 不同 batch_size（`num_workers=8`）

| batch_size | 吞吐 | 每 epoch | GPU 利用率 | 显存（torch 分配 / 进程） |
|---:|---:|---:|---:|---:|
| 1 | 10.5 img/s | 13.82 min | 34% | 1.09 / 1.49 GB |
| 2 | 19.9 img/s | 7.32 min | 52% | 1.62 / 2.12 GB |
| 3 | 29.6 img/s | 4.92 min | 72% | 2.22 / 2.75 GB |
| **4** | **39.7 img/s** | **3.67 min** | **92%** | **2.74 / 3.34 GB** |
| 8 | 43.0 img/s | 3.38 min | 100% | — / 5.94 GB |

> bs8 只比 bs4 快约 8%，显存接近翻倍，推荐 bs4。

### 5.2 不同 num_workers（`batch_size=4`）

| num_workers | 吞吐 | 每 epoch | data_wait 均值 | GPU 利用率 | CPU 占用 |
|---:|---:|---:|---:|---:|---:|
| 0 | 27.4 img/s | 5.31 min | 65.4 ms | 63% | 0.75 核 |
| 1 | 40.4 img/s | 3.60 min | 0.2 ms | 94% | 0.66 核 |
| 2 | 40.5 img/s | 3.59 min | 0.2 ms | 94% | 0.69 核 |
| 3 | 40.2 img/s | 3.62 min | 0.2 ms | 94% | 0.58 核 |
| 4 | 40.3 img/s | 3.61 min | 0.2 ms | 95% | 0.62 核 |
| 6 | 40.7 img/s | 3.57 min | 0.2 ms | 94% | 0.53 核 |
| 8 | 40.3 img/s | 3.61 min | 0.2 ms | 94% | 0.72 核 |
| 12 | 40.2 img/s | 3.62 min | 0.2 ms | 93% | 0.76 核 |

结论：
- **`num_workers=1` 就能跑满**，2–12 之间差异在测量噪声内。
- `num_workers=0` 会因同步解码阻塞 GPU，吞吐下降 32%。
- 整个训练实际只用 **0.5–0.8 个 CPU 核**（12 核机器），4 核实例绰绰有余。

### 5.3 与历史运行对比

| 场景 | 吞吐 | 每 epoch | 备注 |
|---|---:|---:|---|
| 历史运行（2026-10-25，bs1 + identity） | 5.4 img/s | 26.8 min | 日志显示 86°C / ~1500MHz，疑似降频 |
| 当前默认（bs1 + identity，compile） | 9.6 img/s | 15.2 min | 本机当前散热状态下 |
| bs4 + identity | ~30 img/s | ~4.9 min | 保留 identity |
| **bs4 + 关 identity + fused** | **39.7–40.5 img/s** | **~3.6 min** | 推荐 |
| bs8 + 关 identity + fused | 43.0 img/s | 3.4 min | 显存 5.9 GB，收益有限 |

> 相对历史日志约 **7.3 倍**；相对当前默认约 **4.1 倍**。

---

## 6. 已实施的优化（代码改动）

| 文件 | 位置 | 改动 |
|---|---|---|
| `models/cycle_gan_model.py` | 第 78–82 行 | Adam 在 CUDA 下启用 `fused=True`；CPU 自动回退 |
| `train.py` | 第 63–66 行 | `--lr_g_a_scale` 分支的优化器同样启用 fused |
| `data/__init__.py` | 第 54–57 行 | 训练阶段 `batch_size > 1` 时自动 `drop_last`，避免 `torch.compile` 因尾批形状反复编译（每 epoch 最多丢 3 张图） |
| `README.md` | 第 104 行附近、第 190 行起 | 更新加速命令、瓶颈说明、实测吞吐与资源占用表 |

未改动模型结构、损失函数或默认 batch size（保持向后兼容）。

---

## 7. 推荐训练方案与预计时长

### 7.1 从头训练（官方标准 100 + 100，约 12 小时）

```bash
cd /root/autodl-tmp/Chinese_Flower
python train.py --dataroot ./datasets/chinesepainting \
  --name chinesepainting_scratch_bs4 \
  --n_epochs 100 --n_epochs_decay 100 \
  --batch_size 4 --num_workers 2 --identity_off_after 20 \
  --save_epoch_freq 5 --gpu_id 0
```

### 7.2 从头训练（本项目默认 200 + 200，约 24 小时）

把上面的 `--n_epochs 100 --n_epochs_decay 100` 换成 `--n_epochs 200 --n_epochs_decay 200`。

### 7.3 续训现有 `warmstart_short`（从 epoch 2 继续）

```bash
python train.py --dataroot ./datasets/chinesepainting \
  --name chinesepainting_warmstart_short --continue_train \
  --n_epochs 100 --n_epochs_decay 100 \
  --batch_size 4 --num_workers 2 --identity_off_after 2 \
  --save_epoch_freq 5 --gpu_id 0
```

> 续训时**不要**再传 `--pretrained_g_a`，也不要在同一个 `--name` 下从头启动，否则会覆盖 `latest_*`。

### 7.4 时间节点参考（bs4 + 关 identity）

| epoch | 累计时间 |
|---:|---:|
| 20 | ~1.2 h |
| 50 | ~3 h |
| 100 | ~6 h |
| 200（完整衰减） | ~12 h |
| 400（200+200） | ~24 h |

---

## 8. 验证记录

2026-10-26 01:45 用修改后的 `train.py` 跑通 1 个完整 epoch：

```bash
python train.py --dataroot ./datasets/chinesepainting --name verify_bs4 \
  --batch_size 4 --lambda_identity 0 \
  --n_epochs 1 --n_epochs_decay 0 \
  --checkpoints_dir /tmp/speed_verify --save_epoch_freq 1 --gpu_id 0
```

结果：

- `dataset: 8731 images (A: 7283, B: 8731)`，实际训练 8,728 张（`drop_last` 生效）
- `training config: batch_size=4, workers=8, amp=True, compile=True`
- **1 epoch = 221 秒（3.68 分钟）**
- loss 正常：`cycle_A ≈ 2.2`、`cycle_B ≈ 1.6`、`idt = 0`（按 `--lambda_identity 0` 关闭）
- `1_net_G_A/G_B/D_A/D_B.pth`、`latest_*`、`training_state.pth` 全部正常保存
- 产物在 `/tmp/speed_verify/`，未写入项目目录（临时文件，可删除）

---

## 9. 注意事项与风险

1. **不要按 `requirements.txt` 重装依赖**（版本与环境不匹配，Pillow 9.5.0 在 Python 3.12 下无 Linux wheel）。
2. **磁盘**：每 epoch checkpoint 约 108 MB；`--save_epoch_freq 1` 跑 400 epoch 会写约 43 GB，超过当前 41 GiB 可用空间。用 `--save_epoch_freq 5` 或定期清理旧 epoch。
3. **续训语义**：`--continue_train` 只恢复网络权重、epoch 和 total_iters，**不恢复 Adam 优化器状态、GradScaler、RNG**；LR 调度器会按传入的 `n_epochs/n_epochs_decay` 快进到对应 epoch。因此续训与不中断训练不完全等价。
4. **热降频**：如果长时间训练后吞吐明显下降，先查 `clocks.sm` 是否掉到 1500MHz 左右；当前散热状态下未出现。
5. **README 其他章节仍有历史遗留**：多 GPU 的 `--gpu_id all` 无效；`finetune/` 脚本与本项目 checkpoint 命名不兼容；`train.ipynb` 中的 `./dataroot/MoJing-Lab-Datasets` 路径已过期。当前可靠的训练入口只有 `train.py`。
6. **数据**：`trainB` 中 268 个 `.png` 实际是 JPEG 编码（按内容读取，不影响训练）；A/B 数量不同（7,283 vs 8,731），每个 epoch 会重复使用 A 域图片。
7. **推理**：`test.py --name <实验名>` 默认只加载 `latest_net_G_A.pth`；旧 `results_smoke/` 是历史对比图，不是本次产物。

---

## 10. 复现方法

性能数据来自只读基准脚本（真实数据集、真实模型代码、不写项目文件）：

- 数据加载吞吐：遍历 `create_dataset(...)`，`num_workers` 8/12，`batch_size` 1/4
- GPU step 吞吐：`CycleGANModel` + 真实 batch，预热 10–15 步后计时 25 秒
- GPU 指标：`nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,fan.speed --format=csv,noheader,nounits`，1 Hz 采样
- CPU 占用：读取 `/proc/stat`（idle+iowait 与 total 的差值）
- 显存：`torch.cuda.max_memory_allocated()/reserved()` 与 `nvidia-smi memory.used`

如需重新测量，可在 GPU 空闲时按上述方式复跑；正式训练结束后用 `test.py` 出图做主观质量对比。
