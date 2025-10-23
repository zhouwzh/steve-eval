import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_anneal(step, start_value, final_value, start_step, final_step):

    assert start_value >= final_value
    assert start_step <= final_step

    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = 0.5 * (start_value - final_value)
        b = 0.5 * (start_value + final_value)
        progress = (step - start_step) / (final_step - start_step)
        value = a * math.cos(math.pi * progress) + b

    return value


def linear_warmup(step, start_value, final_value, start_step, final_step):

    assert start_value <= final_value
    assert start_step <= final_step

    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = final_value - start_value
        b = start_value
        progress = (step + 1 - start_step) / (final_step - start_step)
        value = a * progress + b

    return value


def gumbel_softmax(logits, tau=1., hard=False, dim=-1):

    eps = torch.finfo(logits.dtype).tiny

    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau

    y_soft = F.softmax(gumbels, dim)

    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           dilation=1, groups=1, bias=True, padding_mode='zeros',
           weight_init='xavier'):
    
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  dilation, groups, bias, padding_mode)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m


class Conv2dBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        
        self.m = conv2d(in_channels, out_channels, kernel_size, stride, padding,
                        bias=True, weight_init='kaiming')
    
    def forward(self, x):
        x = self.m(x)
        return F.relu(x)


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    
    m = nn.Linear(in_features, out_features, bias)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m


def gru_cell(input_size, hidden_size, bias=True):
    
    m = nn.GRUCell(input_size, hidden_size, bias)
    
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    
    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)
    
    return m

def _edge_from_mask(mask_bool, k=3):
    N, T, S, _, H, W = mask_bool.shape
    m = mask_bool.float().view(N*T*S, 1, H, W)
    dil = F.max_pool2d(m, kernel_size=k, stride=1, padding=k//2)  #360, 1, 128, 128
    ero = 1.0 - F.max_pool2d(1.0 - m, kernel_size=k, stride=1, padding=k//2)
    edge = (dil - ero) > 0
    return edge.view(N, T, S, 1, H, W)

def add_border_to_attn_images(attns, 
                              color=(1., 1., 1.),  # RGB [0,1]
                              thickness_px=2, 
                                thr=0.5,
                              use_contour=True,
                              use_bbox=False):
    """
    video: (N,1,3,128,128)
    attns: (N, T, num_slots, 3, 128, 128)
    """
    # import pdb; pdb.set_trace()
    N,T,S,C,H,W = attns.shape
    device = attns.device
    dtype = attns.dtype

    # attns_img = video.squeeze(1) * attns_mask_up + (1. - attns_mask_up)  # (N, num_slots, 3, 128, 128)

    mask_gray = attns.mean(dim=3, keepdim=True)
    mask_bin = (mask_gray >= thr) # (N, T, S, 1, H, W)

    attns_img = attns.clone()

    edge = _edge_from_mask(mask_bin, k=3)
    
    ks = 2*thickness_px - 1 if thickness_px > 1 else 1
    if ks > 1:
        e = edge.float().view(N*T*S, 1, H, W)
        e = torch.nn.functional.max_pool2d(e, kernel_size=ks, stride=1, padding=thickness_px//2)
        edge = (e > 0).view(N, T, S, 1, H, W)
    
    col = torch.tensor(color, dtype=dtype, device=device).view(1,1,1,3,1,1)  # (1,1,1,C,1,1)
    edge_c = edge.expand(N,T,S,C,H,W)   # (B,T,S,C,H,W)
    attns_img = attns_img.clone()
    attns_img[edge_c] = col.expand_as(attns_img)[edge_c]

    return attns_img


