from functools import partial

from einops import rearrange
from timm.models import register_model
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from torch import nn

from .stage import StageTransformer, _cfg


class LinAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_tokens,
        kv_tokens_ratio=4,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_kv = nn.Linear(num_tokens, num_tokens // kv_tokens_ratio)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = rearrange(self.qkv(x), "B N (qkv H C) -> qkv B H C N", qkv=3, H=self.num_heads)
        q, k, v = [
            rearrange(qkv[0], "B H C N -> B H N C"),
            self.proj_kv(qkv[1]),  # B H C K
            rearrange(self.proj_kv(qkv[2]), "B H C K -> B H K C"),
        ]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = rearrange(attn @ v, "B H N C -> B N (H C)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LinformerBlock(nn.Module):
    r"""Linformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        num_tokens = input_resolution[0] * input_resolution[1]
        self.norm1 = norm_layer(dim)
        self.attn = LinAttention(
            dim,
            num_tokens,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            **kwargs,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution},"
            f"num_heads={self.num_heads}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # FIXME: attn
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


@register_model
def stage_tiny_lin_p4(pretrained=False, **kwargs):
    cfg = _cfg(
        patch_size=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), kv_tokens_ratio=8, **kwargs
    )
    model = StageTransformer(LinformerBlock, **cfg)
    return model


@register_model
def stage_tiny_lin_p7(pretrained=False, **kwargs):
    cfg = _cfg(
        patch_size=7, norm_layer=partial(nn.LayerNorm, eps=1e-6), kv_tokens_ratio=8, **kwargs
    )
    model = StageTransformer(LinformerBlock, **cfg)
    return model
