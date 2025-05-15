import math
import torch
import torch.nn.functional as F
from torch import nn

from .kernel.rotary import apply_rotary_emb
#from flash_attn import flash_attn_func
'''
try:
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    print("No fused RMSNorm")
'''
from .rms_norm import RMSNorm


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

def l2_regularization(self, model):
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2) ** 2
    return l2_reg

class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth, # current layer index
        num_heads,
        num_kv_heads=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # arg num_heads set to half of baseline Transformer's num_heads
        # for e.g., to compare with a baseline Transformer with 16 heads, pass in num_heads=8 for DIFF Transformer
        self.num_heads = num_heads
        #self.num_kv_heads = num_heads
        
        # arg num_kv_heads set to half of baseline Transformer's num_kv_heads if use GQA
        # for e.g., to compare with a baseline Transformer with 16 heads and 8 kv_heads, 
        # pass in num_heads=8, num_kv_heads=4 for DIFF Transformer
        # if use MHA, pass in num_kv_heads=None
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        #self.n_rep = 1
        self.n_rep = self.num_heads // self.num_kv_heads
        #print("embed_dim",embed_dim)
        #print("num_heads",num_heads)
        
        self.head_dim = embed_dim // num_heads // 2
        #print("self.head_dim",self.head_dim)
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        #self.in_proj = nn.Linear(embed_dim, 2*embed_dim , bias=False)

        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
        self.attn_dropout = nn.Dropout(p=0.1)
        #self.output_dropout = nn.Dropout(p=0.1)

        #self.q_norm = RMSNorm(embed_dim, eps=1e-5)
        #self.k_norm = RMSNorm(embed_dim // self.n_rep, eps=1e-5)

    
    def forward(
        self,
        query,
        key,
        value,
        rel_pos,
        attn_mask,
    ):
        
        #cos, sin = rel_pos
        q,k,v = [x.transpose(1, 0) for x in (query,key,value)]
        bsz, tgt_len, embed_dim = q.size()
        src_len = tgt_len
        
        #q = self.q_norm(self.q_proj(q))
        #k = self.k_norm(self.k_proj(k))

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        #print("self.num_heads",self.num_heads)
        #print("self.embed_dim",self.embed_dim)
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        #q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        #k = apply_rotary_emb(k, *rel_pos, interleaved=True)



        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        '''
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        '''
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        
        q *= self.scaling
        '''
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        '''

        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        else:  
            attn_mask = attn_mask.view(bsz, 1, 1, src_len).expand(bsz, 2*self.num_heads, 1, src_len)
            #attn_mask = attn_mask.reshape(bsz, 2*self.num_heads, 1, src_len)
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
            #print("attn_mask.shape",attn_mask.shape)
            #print("attn_weights.shape",attn_weights.shape)
            #attn_mask = attn_mask.float()
            #attn_mask = torch.matmul(attn_mask, attn_mask.transpose(-1, -2))
            #attn_mask = attn_mask * float('-inf')  

        
        #offset = 0
        '''
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        '''
        
        #print("attn_mask",attn_mask.shape)
        #attn_weights.nan_to_num_(nan=0.0, posinf=None, neginf=None)

        #print("attn_weights",attn_weights.shape)
        #print("attn_mask",attn_mask.shape)
        
        attn_weights += attn_mask
        attn_weights = torch.nan_to_num(attn_weights)
        #attn_weights.masked_fill_(attn_mask.bool, float('-inf'))
        
           
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        #print("attn_weights.shape",attn_weights.shape)
        # 添加稳定性：避免溢出
        attn_weights = torch.clamp(attn_weights, min=-1e6, max=1e6)

        attn_weights = self.attn_dropout(attn_weights)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        #lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_q1, dim=-1).float()).type_as(q)
        #lambda_2 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_q1, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        #lambda_full = self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        #print("attn_weights.shape_2",attn_weights.shape)
        
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        attn = attn.transpose(0, 1)
        attn = self.out_proj(attn)
        #attn = self.output_dropout(attn)  # final output dropout
        return attn
