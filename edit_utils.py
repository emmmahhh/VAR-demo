import torch
import torch.nn.functional as F
from typing import List, Optional, Union
from models.var import sample_with_top_k_top_p_, gumbel_softmax_with_rng
import PIL.Image as PImage

# 从 models.var 导入所需类（仅用于类型提示）
from models.var import AdaLNSelfAttn


def get_edit_mask(patch_nums: List[int], y0: float, x0: float, y1: float, x1: float, device, inpainting: bool = True) -> torch.Tensor:
    """
    生成编辑掩码，用于指定哪些 token 需要保留（1）或生成（0）。
    """
    ph, pw = patch_nums[-1], patch_nums[-1]
    edit_mask = torch.zeros(ph, pw, device=device)
    edit_mask[round(y0 * ph):round(y1 * ph), round(x0 * pw):round(x1 * pw)] = 1
    if inpainting:
        edit_mask = 1 - edit_mask
    return edit_mask


def replace_embedding(edit_mask: torch.Tensor, h_BChw: torch.Tensor, gt_BChw: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    """
    根据掩码替换 token 嵌入。
    """
    B = h_BChw.shape[0]
    h, w = edit_mask.shape[-2:]
    if edit_mask.ndim == 2:
        edit_mask = edit_mask.unsqueeze(0).expand(B, h, w)
    force_gt_B1hw = F.interpolate(edit_mask.unsqueeze(1).to(dtype=torch.float, device=gt_BChw.device), size=(ph, pw), mode='bilinear', align_corners=False).gt(0.5).int()
    if ph * pw <= 3:
        force_gt_B1hw.fill_(1)
    return gt_BChw * force_gt_B1hw + h_BChw * (1 - force_gt_B1hw)


def autoregressive_infer_cfg_with_mask(
    var,  # VAR 模型实例
    B: int, label_B: Optional[Union[int, torch.LongTensor]],
    g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
    more_smooth=False,
    input_img_tokens: Optional[List[torch.Tensor]] = None,
    edit_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    支持掩码的自回归推理函数，用于零样本编辑。
    此函数与 VAR 类中的 autoregressive_infer_cfg 类似，但增加了 input_img_tokens 和 edit_mask 参数。
    """
    # 将 self 替换为 var
    if g_seed is None:
        rng = None
    else:
        var.rng.manual_seed(g_seed)
        rng = var.rng

    if label_B is None:
        label_B = torch.multinomial(var.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
    elif isinstance(label_B, int):
        label_B = torch.full((B,), fill_value=var.num_classes if label_B < 0 else label_B, device=var.lvl_1L.device)

    sos = cond_BD = var.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=var.num_classes)), dim=0))

    lvl_pos = var.lvl_embed(var.lvl_1L) + var.pos_1LC
    next_token_map = sos.unsqueeze(1).expand(2 * B, var.first_l, -1) + var.pos_start.expand(2 * B, var.first_l, -1) + lvl_pos[:, :var.first_l]

    cur_L = 0
    f_hat = sos.new_zeros(B, var.Cvae, var.patch_nums[-1], var.patch_nums[-1])

    for b in var.blocks:
        b.attn.kv_caching(True)

    for si, pn in enumerate(var.patch_nums):
        ratio = si / var.num_stages_minus_1
        cur_L += pn * pn
        cond_BD_or_gss = var.shared_ada_lin(cond_BD)
        x = next_token_map
        for b in var.blocks:
            x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
        logits_BlV = var.get_logits(x, cond_BD)

        t = cfg * ratio
        logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

        idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
        if not more_smooth:
            h_BChw = var.vae_quant_proxy[0].embedding(idx_Bl)
        else:
            gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
            h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ var.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

        h_BChw = h_BChw.transpose_(1, 2).reshape(B, var.Cvae, pn, pn)

        if edit_mask is not None and input_img_tokens is not None:
            gt_BChw = var.vae_quant_proxy[0].embedding(input_img_tokens[si]).transpose_(1, 2).reshape(B, var.Cvae, pn, pn)
            h_BChw = replace_embedding(edit_mask, h_BChw, gt_BChw, pn, pn)

        f_hat, next_token_map = var.vae_quant_proxy[0].get_next_autoregressive_input(si, len(var.patch_nums), f_hat, h_BChw)
        if si != var.num_stages_minus_1:
            next_token_map = next_token_map.view(B, var.Cvae, -1).transpose(1, 2)
            next_token_map = var.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + var.patch_nums[si+1] ** 2]
            next_token_map = next_token_map.repeat(2, 1, 1)

    for b in var.blocks:
        b.attn.kv_caching(False)

    return var.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)