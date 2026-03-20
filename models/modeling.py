# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs

from .modeling_resnet import ResNetV2


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


@dataclass(frozen=True)
class AttentionResidualsConfig:
    mode: str = "none"  # "none" | "full" | "block"
    num_blocks: int = 8
    block_size: Optional[int] = None  # transformer layers per block; overrides num_blocks when set
    eps: float = 1e-6
    log_stats: bool = False
    collect_alphas: bool = False  # when vis=True, store alphas for analysis (cls token, batch-mean)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        return x_norm.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


class AttentionResidualsMixer(nn.Module):
    """
    Depth-wise softmax attention over a list of source representations.

    sources: list of tensors [B, T, D]
    returns: mixed tensor [B, T, D]
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.proj = nn.Linear(hidden_size, 1, bias=False)  # pseudo-query w_l (stored as weight)
        nn.init.zeros_(self.proj.weight)

    def forward(
        self,
        sources: List[torch.Tensor],
        collect_alpha: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if len(sources) == 0:
            raise ValueError("AttentionResidualsMixer requires at least one source tensor.")
        v = torch.stack(sources, dim=0)  # [N, B, T, D]
        k = self.norm(v)
        w = self.proj.weight.squeeze(0)  # [D]
        logits = torch.einsum("d, n b t d -> n b t", w, k)  # [N, B, T]
        alpha = logits.softmax(dim=0)  # [N, B, T]
        mixed = torch.einsum("n b t, n b t d -> b t d", alpha, v.float()).to(dtype=v.dtype)  # [B, T, D]

        if not collect_alpha:
            return mixed, None

        # Store a compact alpha summary for analysis: CLS token only, averaged across batch.
        # Shape: [N]
        alpha_summary = alpha[:, :, 0].mean(dim=1)
        return mixed, alpha_summary


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis, attnres_cfg: Optional[AttentionResidualsConfig] = None):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

        self.attnres_cfg = attnres_cfg
        self.use_attnres = attnres_cfg is not None and attnres_cfg.mode in ("full", "block")
        if self.use_attnres:
            self.attnres_attn = AttentionResidualsMixer(config.hidden_size, eps=attnres_cfg.eps)
            self.attnres_mlp = AttentionResidualsMixer(config.hidden_size, eps=attnres_cfg.eps)

    def forward(self, x):
        if self.use_attnres:
            raise RuntimeError("Block.forward(x) is not used when Attention Residuals is enabled.")

        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def forward_with_attnres(
        self,
        hidden_states: torch.Tensor,
        blocks: List[torch.Tensor],
        partial_block: torch.Tensor,
        include_partial_for_attn: bool,
        collect_alphas: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Apply Block AttnRes before attention.
        sources_attn = blocks + ([partial_block] if include_partial_for_attn else [])
        h_attn, alpha_attn = self.attnres_attn(sources_attn, collect_alpha=collect_alphas)
        attn_out, weights = self.attn(self.attention_norm(h_attn))
        partial_block = partial_block + attn_out

        # Apply Block AttnRes before MLP (always includes partial sum, per paper pseudo-code).
        sources_mlp = blocks + [partial_block]
        h_mlp, alpha_mlp = self.attnres_mlp(sources_mlp, collect_alpha=collect_alphas)
        mlp_out = self.ffn(self.ffn_norm(h_mlp))
        partial_block = partial_block + mlp_out

        hidden_states = partial_block
        return hidden_states, weights, alpha_attn, alpha_mlp

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis, attnres_cfg: Optional[AttentionResidualsConfig] = None):
        super(Encoder, self).__init__()
        self.vis = vis
        self.attnres_cfg = attnres_cfg
        self.use_attnres = attnres_cfg is not None and attnres_cfg.mode in ("full", "block")
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis, attnres_cfg=attnres_cfg)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []

        # Exposed analysis artifacts (populated when configured).
        self.last_attnres_alphas: Optional[List[dict]] = None
        self.last_attnres_stats: Optional[dict] = None

        if not self.use_attnres:
            for layer_block in self.layer:
                hidden_states, weights = layer_block(hidden_states)
                if self.vis:
                    attn_weights.append(weights)
            encoded = self.encoder_norm(hidden_states)
            return encoded, attn_weights

        assert self.attnres_cfg is not None
        num_layers = len(self.layer)
        if self.attnres_cfg.mode == "full":
            block_size = 1
        else:
            if self.attnres_cfg.block_size is not None:
                block_size = max(1, int(self.attnres_cfg.block_size))
            else:
                num_blocks = max(1, int(self.attnres_cfg.num_blocks))
                block_size = max(1, math.ceil(num_layers / num_blocks))

        blocks: List[torch.Tensor] = [hidden_states]  # b0 = token embedding
        partial_block = hidden_states

        collect_alphas = bool(self.vis and self.attnres_cfg.collect_alphas)
        alphas: List[dict] = []
        stats_norms: List[torch.Tensor] = []

        for layer_idx, layer_block in enumerate(self.layer):
            is_block_start = (layer_idx % block_size) == 0
            if is_block_start:
                partial_block = hidden_states

            include_partial_for_attn = not is_block_start
            hidden_states, weights, alpha_attn, alpha_mlp = layer_block.forward_with_attnres(
                hidden_states=hidden_states,
                blocks=blocks,
                partial_block=partial_block,
                include_partial_for_attn=include_partial_for_attn,
                collect_alphas=collect_alphas,
            )
            partial_block = hidden_states

            if self.vis:
                attn_weights.append(weights)
            if collect_alphas:
                alphas.append(
                    {
                        "layer": layer_idx,
                        "block_size": block_size,
                        "num_blocks_so_far": len(blocks),
                        "alpha_attn": alpha_attn,
                        "alpha_mlp": alpha_mlp,
                    }
                )
            if self.attnres_cfg.log_stats:
                with torch.no_grad():
                    # Mean RMS across tokens and batch.
                    rms = hidden_states.float().pow(2).mean(dim=-1).sqrt().mean()
                    stats_norms.append(rms.detach())

            is_block_end = ((layer_idx + 1) % block_size) == 0 or (layer_idx == num_layers - 1)
            if is_block_end:
                blocks.append(partial_block)

        if collect_alphas:
            self.last_attnres_alphas = alphas
        if self.attnres_cfg.log_stats:
            self.last_attnres_stats = {"block_size": block_size, "hidden_rms_by_layer": stats_norms}

        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, attnres_cfg: Optional[AttentionResidualsConfig] = None):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis, attnres_cfg=attnres_cfg)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(
        self,
        config,
        img_size=224,
        num_classes=21843,
        zero_head=False,
        vis=False,
        attnres_cfg: Optional[AttentionResidualsConfig] = None,
    ):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis, attnres_cfg=attnres_cfg)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
