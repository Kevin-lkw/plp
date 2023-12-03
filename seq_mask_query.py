import torch
from torch import nn
from einops import rearrange


class UpSampling(nn.Module):
    def __init__(
            self,
            out_ch,
            scale_factor=4,
            mode='nearest'
    ):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(
            self,
            x
    ):
        x = self.up(x)
        x = self.conv(x)

        return x


class LinearAttention(nn.Module):
    def __init__(
            self,
            out_ch,
            heads=4,
            dim_head=32
    ):
        super().__init__()

        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(out_ch, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, out_ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)

        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)

        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)

        return self.to_out(out)


class Block(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(
            self,
            x,
            scale_shift=None
    ):
        x = self.conv(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)

        return x


class ResnetBlock(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            t_emb_dim=None
    ):
        super().__init__()

        if t_emb_dim is not None:
            self.t_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_ch * 2)
                )
        else:
            self.t_mlp = None

        self.block1 = Block(in_ch, out_ch)
        self.block2 = Block(out_ch, out_ch)

        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(
            self,
            x,
            t_emb=None
    ):
        scale_shift = None
        if self.t_mlp is not None and t_emb is not None:
            t_emb = self.t_mlp(t_emb)
            t_emb = rearrange(t_emb, "b c -> b c 1 1")
            scale_shift = t_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


class Mask(nn.Module):
    def __init__(
        self,
        in_ch=1280,
        out_ch=1,
        init_ch=256,
        t_emb_dim=2048,
        ch_div=(0, 2, 4, 6, 8),  # channel divider
        attn_idx=(1, 2),  # attention index
    ):
        super().__init__()
        self.chs = [init_ch // (2 ** div) for div in ch_div]
        self.in_out_chs = list(zip(self.chs[:-1], self.chs[1:]))

        self.conv_in = nn.Conv2d(in_ch, self.chs[0], kernel_size=1)

        self.mask_ups = nn.ModuleList([])
        for idx, (ch_in, ch_out) in enumerate(self.in_out_chs):
            self.mask_ups.append(
                nn.Sequential(
                    ResnetBlock(ch_in, ch_out, t_emb_dim),
                    ResnetBlock(ch_out, ch_out, t_emb_dim),
                    LinearAttention(ch_out) if idx in attn_idx else nn.Identity(),
                    UpSampling(ch_out) if idx != len(self.in_out_chs) - 1 else nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
                )
            )

        self.conv_out = nn.Conv2d(self.chs[-1], out_ch, kernel_size=3, padding=1)

        self.act = nn.Sigmoid()

    def forward(
        self,
        x,
        t_emb=None
    ):
        # print(x.shape)

        x = self.conv_in(x)
        # print(x.shape)

        for block1, block2, attn, up in self.mask_ups:
            x = block1(x, t_emb)
            # print(x.shape)
            x = block2(x, t_emb)
            # print(x.shape)
            x = attn(x)
            x = up(x)
            # print(x.shape)

        x = self.conv_out(x)

        x = self.act(x)
        # print(x.shape)

        return x


if __name__ == "__main__":
    mask = Mask().to('cuda')

    x = torch.randn(4, 1280, 8, 8).to('cuda')

    output = mask(x)
    loss = nn.functional.mse_loss(output, torch.zeros_like(output))

    loss.backward()

