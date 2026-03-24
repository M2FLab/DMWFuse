import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from einops import rearrange

class WaveletFeatureExtractor(nn.Module):
    def __init__(self, wavelet_type='haar'):
        super(WaveletFeatureExtractor, self).__init__()
        self.wavelet_type = wavelet_type

    def forward(self, x):
        coeffs = pywt.dwt2(x.detach().cpu().numpy(), self.wavelet_type)
        LL, (LH, HL, HH) = coeffs
        LL = torch.from_numpy(LL).to(x.device)
        LH = torch.from_numpy(LH).to(x.device)
        HL = torch.from_numpy(HL).to(x.device)
        HH = torch.from_numpy(HH).to(x.device)
        high_freq = torch.cat([LH.unsqueeze(1), HL.unsqueeze(1), HH.unsqueeze(1)], dim=1)
        return LL, high_freq


class WaveletAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=2, window_size=8):
        super(WaveletAttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, wavelet_features):
        B, C, H, W = x.shape
        
        x = rearrange(x, 'b c (h wh) (w ww) -> b (h w) (wh ww) c', wh=self.window_size, ww=self.window_size)
        wavelet_features = rearrange(wavelet_features, 'b c (h wh) (w ww) -> b (h w) (wh ww) c', wh=self.window_size, ww=self.window_size)
        
        def _window_attention(q, k, v):
            q = q.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = k.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
            v = v.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
            return out
        
        q = self.q(x)
        k = self.k(wavelet_features)
        v = self.v(wavelet_features)
        
        out = torch.utils.checkpoint.checkpoint(_window_attention, q, k, v)
        
        out = rearrange(out, 'b (h w) (wh ww) c -> b c (h wh) (w ww)', 
                         h=H//self.window_size, w=W//self.window_size, 
                         wh=self.window_size, ww=self.window_size)
        out = self.proj(out)
        return out