import torch
import torch.nn as nn
import torch.nn.functional as f

class SpatialInjector(nn.Module):
    def __init__(self, vit_dim=1024, cnn_dims=[40, 80, 160, 320], target_grid=64):
        """
        Bơm thông tin không gian từ CNN vào ViT một cách tối ưu bộ nhớ.
        """
        super().__init__()
        # Sử dụng Adaptive Pooling để nén các feature map lớn (1/4, 1/8) xuống 
        # kích thước an toàn (64x64) để chống tràn VRAM khi tính Attention.
        self.cnn_projections = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((target_grid, target_grid)),
                nn.Conv2d(dim, vit_dim, kernel_size=1)
            ) for dim in cnn_dims
        ])
        
        self.norm = nn.LayerNorm(vit_dim)
        self.scale = vit_dim ** -0.5

    def forward(self, vit_tokens, cnn_features):
        flattened_cnn = []
        
        for i, feat in enumerate(cnn_features):
            # feat_proj: [B, 1024, 64, 64]
            feat_proj = self.cnn_projections[i](feat)
            b, c, h, w = feat_proj.shape
            
            # Ép phẳng: [B, 4096, 1024]
            feat_flat = feat_proj.view(b, c, -1).transpose(1, 2)
            flattened_cnn.append(feat_flat)
            
        # Tổng tokens bây giờ chỉ là 4 * 4096 = 16,384 tokens (Cực kỳ an toàn)
        s_flat = torch.cat(flattened_cnn, dim=1) 

        # Tính toán Cross-Attention
        '''
        attn_scores = torch.matmul(vit_tokens, s_flat.transpose(-2, -1)) * self.scale
        attn_weights = f.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, s_flat)
        '''

        # Flash attention: Sử dụng torch.nn.functional.scaled_dot_product_attention để tận dụng tối đa hiệu suất GPU.
        attn_output = f.scaled_dot_product_attention(vit_tokens, s_flat, s_flat, dropout_p=0.0, is_causal=False)

        return self.norm(vit_tokens + attn_output)