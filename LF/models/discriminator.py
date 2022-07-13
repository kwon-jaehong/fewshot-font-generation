"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from functools import partial
import torch
import torch.nn as nn
from base.modules import ResBlock, ConvBlock, w_norm_dispatch, activ_dispatch


class ProjectionDiscriminator(nn.Module):
    """ Multi-task discriminator """
    def __init__(self, C, n_fonts, n_chars, w_norm='spectral', activ='none'):
        super().__init__()

        self.activ = activ_dispatch(activ)()
        w_norm = w_norm_dispatch(w_norm)
        self.font_emb = w_norm(nn.Embedding(n_fonts, C))
        self.char_emb = w_norm(nn.Embedding(n_chars, C))

    def forward(self, x, font_indice, char_indice):
        x = self.activ(x)
        font_emb = self.font_emb(font_indice)
        char_emb = self.char_emb(char_indice)

        font_out = torch.einsum('bchw,bc->bhw', x.float(), font_emb.float()).unsqueeze(1)
        char_out = torch.einsum('bchw,bc->bhw', x.float(), char_emb.float()).unsqueeze(1)

        return [font_out, char_out]


class Discriminator(nn.Module):
    """
    spectral norm + ResBlock + Multi-task Discriminator (No patchGAN)
    """
    def __init__(self, n_fonts, n_chars):
        super().__init__()
        ConvBlk = partial(ConvBlock, w_norm="spectral", activ="relu", pad_type="zero")
        ResBlk = partial(ResBlock, w_norm="spectral", activ="relu", pad_type="zero", scale_var=False)

        C = 32
        self.feats = nn.ModuleList([
            ConvBlk(1, C, stride=2, activ='none'),  # 64x64 (stirde==2)
            ResBlk(C*1, C*2, downsample=True),    # 32x32
            ResBlk(C*2, C*4, downsample=True),    # 16x16
            ResBlk(C*4, C*8, downsample=True),    # 8x8
            ResBlk(C*8, C*16, downsample=False),  # 8x8
            ResBlk(C*16, C*16, downsample=False),  # 8x8
        ])

        gap_activ = activ_dispatch("relu")
        self.gap = nn.Sequential(
            gap_activ(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.projD = ProjectionDiscriminator(C*16, n_fonts, n_chars, w_norm="spectral")
        # ProjectionDiscriminator(
        # (activ): Identity()
        # (font_emb): Embedding(1393, 512)
        # (char_emb): Embedding(2203, 512)
        # )
        # 폰트와, 문자를 임베딩 연산

    def forward(self, x, font_indice, char_indice, out_feats='none'):
        assert out_feats in {'none', 'all'}
        feats = []
        for layer in self.feats:
            x = layer(x)
            feats.append(x)

        x = self.gap(x)  # final features
        
        ret = self.projD(x, font_indice, char_indice)
        # ret = [tensor([[[[0.0203]]]...ackward0>), tensor([[[[ 0.0043]]...ackward0>)]
        # projd값은 폰트,문자 라벨의 임베딩값을 리턴함
        
        
        ## out_feats = 판별자에서 컨브블럭,레즈블럭에서 나온 특징들을 모두 가져옴
        if out_feats == 'all':
            ret += feats

        ret = tuple(map(lambda i: i.cuda(), ret))
        return ret
