# Attention Residuals in Vision Transformer

This repository implements **Attention Residuals (AttnRes)**, an extension to the Vision Transformer (ViT) architecture based on the paper *Attention Residuals: Learning to Skip Layers via Attention Feedback* (2026).

## 1. Vision Transformer (ViT) 简介

Vision Transformer 将图像分割成固定大小的 patch（如 16×16），每个 patch 展平后作为 token 输入标准 Transformer 编码器。

**标准 Transformer 编码器层**：
```
h_l = TransformerBlock(h_{l-1})
    = LayerNorm(MSA(h_{l-1}) + h_{l-1})      # Multi-Head Self-Attention + 残差连接
    = LayerNorm(MLP(LayerNorm(...)) + ...)   # FFN + 残差连接
```

其中 MSA (Multi-Head Self-Attention) 的计算：
```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

---

## 2. Attention Residuals 核心思想

### 2.1 标准残差的局限

标准残差连接：`h_l = f(h_{l-1}) + h_{l-1}`

这允许梯度直接回传，但 **f(h_{l-1}) 的计算开销并未减少**。深层网络仍然需要逐层计算所有 Transformer blocks。

### 2.2 Attention Residuals 解决方案

**核心公式**：
```
h_l = AttnResBlock(h_{l-1}) = α · softmax(α) · Attn(h_{l-1}) + h_{l-1}
```

关键改进：**将 Attention 的输出通过 softmax 加权后加回输入**

### 2.3 Block Attention Residuals

将 L 个 Transformer 层划分为 B 个 blocks（例如 B=8）：

```
对于第 b 个 block (包含多个层)：
  sources = [h_{b*block_size}, h_{b*block_size+1}, ..., h_{b*block_size+block_size-1}]
  
  对每个位置 t 的 query：
    α_t = softmax(W_α · concat([h_i[t] for h_i in sources]))  # 跨深度的注意力权重
    α_t_attn = α_t · [Attn_i(h_i[t]) for i in sources]        # 加权聚合
  
  输出：
    h_{b*block_size+k}[t] = α_t_attn · attn_weight_k + h_{b*block_size}[t]
```

其中 `attn_weight_k` 是第 k 层 attention 的可学习权重。

---

## 3. 实现细节

### 3.1 配置参数

```python
from models.modeling import AttentionResidualsConfig, VisionTransformer, CONFIGS

cfg = CONFIGS["ViT-B_16"]
attnres = AttentionResidualsConfig(
    mode="block",           # "full" 或 "block"
    num_blocks=8,           # block 模式下的 block 数量
    block_size=None,        # 每个 block 的层数（自动计算）
    eps=1e-6,               # 数值稳定性参数
    collect_alphas=False,   # 是否收集 α 权重用于分析
)
model = VisionTransformer(cfg, img_size=224, num_classes=10, 
                          zero_head=True, vis=True, attnres_cfg=attnres)
```

### 3.2 模式说明

| 模式 | 描述 | α 权重含义 |
|------|------|-----------|
| `none` | 标准 ViT（基线） | 无 |
| `full` | 每个 Transformer 层作为独立 block | 衡量单层 attention 的贡献度 |
| `block` | 多层划分为一个 block | 衡量深度方向的信息混合 |

### 3.3 关键代码路径

- **Attention Residuals 计算**: `models/modeling.py` → `TransformerBlock.forward_with_attnres()`
- **Block 混合**: `models/modeling.py` → `AttentionResidualsBlock.forward()`
- **α 收集**: `models/modeling.py` → `TransformerEncoder.last_attnres_alphas`

---

## 4. 训练命令

```bash
# 基线（标准残差）
python train.py --name cifar10_baseline --dataset cifar10 \
    --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz

# Full AttnRes（每层独立 block）
python train.py --name cifar10_attnres_full --dataset cifar10 \
    --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
    --attnres full

# Block AttnRes（~8 个 blocks）
python train.py --name cifar10_attnres_block --dataset cifar10 \
    --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
    --attnres block --attnres_num_blocks 8
```

---

## 5. 分析 α 权重

收集训练过程中的深度混合权重进行分析：

```python
import torch
from models.modeling import VisionTransformer, CONFIGS, AttentionResidualsConfig

cfg = CONFIGS["ViT-B_16"]
attnres = AttentionResidualsConfig(mode="block", num_blocks=8, collect_alphas=True)
model = VisionTransformer(cfg, img_size=224, num_classes=10, 
                          zero_head=True, vis=True, attnres_cfg=attnres).eval()

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    _logits, _attn = model(x)

# 获取每层的 α 权重（CLS token，batch 平均）
alphas = model.transformer.encoder.last_attnres_alphas
# alphas shape: [num_blocks, block_size, num_heads] 或 [num_layers, num_heads]
```

---

## 6. 参考

- **ATTENTION RESIDUALS**  [ATTENTION RESIDUALS](https://github.com/MoonshotAI/Attention-Residuals)

- **ViT 原论文**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- **Google ViT**: [github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)
- **Vit pytorch**:[https://github.com/jeonsworld/ViT-pytorch]
