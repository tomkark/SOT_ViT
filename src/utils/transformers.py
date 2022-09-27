import torch
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init
import torch.nn.functional as F
from .stochastic_depth import DropPath
from SOT import SOT
from matplotlib import pyplot as plt
from time import time
import numpy as np
import torchvision
import cv2
import math


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = (number_bins - 1) * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


class Attention(Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1, **kwargs):
        super().__init__()
        print(kwargs)
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)
        self.SOT = SOT(ot_reg=kwargs.get('ot', False))
        self.withSOT = kwargs.get('withSOT', False)
        self.plot = kwargs.get('plot', False)
        self.qk = kwargs.get('qk', False)
        self.saved_file = kwargs.get('saved_file', False)
        self.mean, self.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        self.first = True


    @staticmethod
    def plot_pair(axes, j, title_hist, p, **kwargs):
        pl = axes[0][j]
        if kwargs.get('noRange', False):
            pl.hist(np.log2(p.flatten() + 1), bins=1000)
        else:
            pl.hist(p.flatten(), bins=1000, range=(-1, 1))
        pl.set_title(title_hist)
        pl = axes[1][j]
        i = pl.imshow(p, cmap='hot', interpolation='nearest')
        plt.colorbar(i, ax=pl)

    @staticmethod
    def plot_test(self, attn_no_sot, attn_sot, random_index):
        self.first = False
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 10))
        fig.suptitle('Attention Comparison', fontsize=16)
        p = attn_no_sot[random_index][1]
        # plot the distribution of the attention weights on the left plot
        self.plot_pair(axes, 0, '(Pre-Softmax) Original Attention', p.detach().cpu().numpy())
        attn_no_sot_1 = attn_no_sot * self.scale
        attn_no_sot_1 = attn_no_sot_1.softmax(dim=-1)
        attn_no_sot_1 = self.attn_drop(attn_no_sot_1)
        p = attn_no_sot_1[random_index][1]
        self.plot_pair(axes, 1, '(Post-Softmax) Original Attention Weights', p.detach().cpu().numpy(), noRange=True)
        p = attn_sot[random_index][1]
        self.plot_pair(axes, 2, '(Pre-Softmax) SOT Attention Weights', p.detach().cpu().numpy())
        p = attn_sot[random_index][1]
        self.plot_pair(axes, 3, '(Post-Softmax) SOT Attention Weights', p.detach().cpu().numpy(), noRange=True)
        plt.show()
        if isinstance(self.saved_file, bool):
            print("Need to pass first file name to load  the plot")
            return
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 15))
        fig.suptitle('Patches', fontsize=16)
        loaded = torchvision.utils.make_grid(torch.load(self.saved_file))
        # Inverting the normalization
        loaded = loaded.permute(1, 2, 0).mul(torch.tensor(self.std))
        loaded += torch.tensor(self.mean)
        loaded = loaded.detach().cpu().numpy()
        avi1 = cv2.rectangle(cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY), (15, 15), (21, 21), (0, 0, 0), 1)
        axes[0][0].imshow(avi1, cmap='gray')
        avi2 = cv2.rectangle(cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY), (11, 23), (16, 28), (0, 0, 0), 1)
        axes[1][0].imshow(avi2, cmap='gray')
        avi3 = cv2.rectangle(cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY), (11, 3), (16, 8), (0, 0, 0), 1)
        axes[2][0].imshow(avi3, cmap='gray')

        patch_heatmap = torch.zeros(32, 32, device=attn_sot.device)
        patch_heatmap2 = torch.zeros(32, 32, device=attn_sot.device)
        patch_heatmap3 = torch.zeros(32, 32, device=attn_sot.device)
        cloned = attn_sot.clone() if self.withSOT else attn_no_sot.clone()
        for i in range(32):
            for j in range(32):
                patch_heatmap[i, j] = cloned[0, 0, 36, math.floor((i / 4)) * 8 + math.floor(j / 4)]
                patch_heatmap2[i, j] = cloned[0, 0, 51, math.floor((i / 4)) * 8 + math.floor(j / 4)]
                patch_heatmap3[i, j] = cloned[0, 0, 11, math.floor((i / 4)) * 8 + math.floor(j / 4)]

        patch_heatmap = patch_heatmap.detach().cpu().numpy()
        patch_heatmap2 = patch_heatmap2.detach().cpu().numpy()
        patch_heatmap3 = patch_heatmap3.detach().cpu().numpy()

        patch_heatmap, _ = image_histogram_equalization(patch_heatmap)
        patch_heatmap2, _ = image_histogram_equalization(patch_heatmap2)
        patch_heatmap3, _ = image_histogram_equalization(patch_heatmap3)

        i = axes[0][1].imshow(patch_heatmap / 256, cmap='hot', interpolation='nearest')
        plt.colorbar(i, ax=axes[0][1])
        i = axes[1][1].imshow(patch_heatmap2 / 256, cmap='hot', interpolation='nearest')
        plt.colorbar(i, ax=axes[1][1])
        i = axes[2][1].imshow(patch_heatmap3 / 256, cmap='hot', interpolation='nearest')
        plt.colorbar(i, ax=axes[2][1])
        plt.show()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.plot or not self.withSOT:
            attn = (q @ k.transpose(-2, -1))
            if self.plot:
                attn_no_sot = attn.clone()
        if self.plot or self.withSOT:
            attn = torch.zeros(q.shape[0], q.shape[1], q.shape[2], q.shape[2], device=v.device)
            if not self.qk:
                for j in range(attn.shape[1]):
                    attn[:, j, :, :] = self.SOT(q[:, j, :, :])
            else:
                for j in range(attn.shape[1]):
                    attn[:, j, :, :] = self.SOT(q[:, j, :, :], k[:, j, :, :])
            if self.plot:
                attn_sot = attn.clone()
        if self.plot and self.first:
            self.plot_test(self, attn_no_sot, attn_sot, torch.randint(0, B, (1,)).item())
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MaskedAttention(Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max
            assert mask.shape[-1] == attn.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn.masked_fill_(~mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1, **kwargs):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout, **kwargs)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class MaskedTransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(MaskedTransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = MaskedAttention(dim=d_model, num_heads=nhead,
                                         attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, mask=None, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src), mask))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class TransformerClassifier(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='sine',
                 sequence_length=None,
                 *args, **kwargs):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i], **kwargs)
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class MaskedTransformerClassifier(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='sine',
                 seq_len=None,
                 *args, **kwargs):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.seq_pool = seq_pool

        assert seq_len is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            seq_len += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                seq_len += 1  # padding idx
                self.positional_emb = Parameter(torch.zeros(1, seq_len, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(seq_len,
                                                                          embedding_dim,
                                                                          padding_idx=True),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            MaskedTransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                          dim_feedforward=dim_feedforward, dropout=dropout,
                                          attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x, mask=None):
        if self.positional_emb is None and x.size(1) < self.seq_len:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            if mask is not None:
                mask = torch.cat([torch.ones(size=(mask.shape[0], 1), device=mask.device), mask.float()], dim=1)
                mask = (mask > 0)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x, mask=mask)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim, padding_idx=False):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(0)
        if padding_idx:
            return torch.cat([torch.zeros((1, 1, dim)), pe], dim=1)
        return pe
