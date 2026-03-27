import sys
import os
sys.path.append(".")
import torch
import streamlit as st
from PIL import Image
import numpy as np
from models import build_vae_var
from edit_utils import get_edit_mask, autoregressive_infer_cfg_with_mask
import torchvision.transforms as transforms
from utils.data import normalize_01_into_pm1

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 缓存模型加载
@st.cache_resource
def load_models():
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=16, shared_aln=False,
    )
    vae_ckpt = 'pretrained_models/vae_ch160v4096z32.pth'
    var_ckpt = 'pretrained_models/var_d16.pth'
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    return vae, var

# 图像生成函数（基于类别）
def generate_image(category, cfg=1.5):
    class_map = {"cat": 281, "dog": 207, "tiger": 292, "bird": 88, "car": 817, "flower": 985}
    label = class_map.get(category, 0)
    label_B = torch.tensor([label], device=device)
    with torch.no_grad():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16):
            recon = var.autoregressive_infer_cfg(B=1, label_B=label_B, cfg=cfg, top_k=900, top_p=0.96, g_seed=0, more_smooth=False)
    img = recon[0].clamp(0,1).cpu().permute(1,2,0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

st.title("VAR 图像生成与编辑演示")
st.markdown("基于 VAR 模型 (Visual AutoRegressive)，NeurIPS 2024 Best Paper")

# 加载模型
with st.spinner("正在加载模型..."):
    vae, var = load_models()
st.success("模型加载完成！")

# 选项卡
tab1, tab2 = st.tabs(["图像生成", "局部补全"])

with tab1:
    st.header("根据类别生成图像")
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("选择类别", ["cat", "dog", "tiger", "bird", "car", "flower"])
        cfg_val = st.slider("CFG 引导强度", 1.0, 3.0, 1.5, 0.1)
    with col2:
        if st.button("生成图像"):
            with st.spinner("生成中..."):
                img = generate_image(category, cfg_val)
                st.image(img, caption=f"{category} (cfg={cfg_val})", use_column_width=True)


with tab2:
    st.header("局部补全")
    st.markdown("上传一张图像，用滑块选择要补全的矩形区域，右侧会实时显示矩形位置。确认后点击“执行补全”。")

    img_file = st.file_uploader("上传图像", type=["png", "jpg", "jpeg"], key="inpaint_img")
    if img_file is not None:
        original = Image.open(img_file).convert('RGB')
        # 显示原始图像（用于预览矩形）
        st.image(original, caption="原始图像", use_column_width=True)

        # 矩形掩码参数（归一化坐标）
        col1, col2 = st.columns(2)
        with col1:
            y0 = st.slider("上边界 (0~1)", 0.0, 1.0, 0.3, 0.01)
            x0 = st.slider("左边界 (0~1)", 0.0, 1.0, 0.3, 0.01)
        with col2:
            y1 = st.slider("下边界 (0~1)", 0.0, 1.0, 0.7, 0.01)
            x1 = st.slider("右边界 (0~1)", 0.0, 1.0, 0.7, 0.01)

        # 实时显示矩形框在原图上的位置
        import PIL.ImageDraw as ImageDraw
        preview = original.copy()
        draw = ImageDraw.Draw(preview)
        # 计算像素坐标
        img_w, img_h = original.size
        left = int(x0 * img_w)
        top = int(y0 * img_h)
        right = int(x1 * img_w)
        bottom = int(y1 * img_h)
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        st.image(preview, caption="矩形区域预览（红色框）", use_column_width=True)

        cfg_inp = st.slider("CFG 引导强度", 1.0, 3.0, 1.5, 0.1, key="inp_cfg")
        if st.button("执行补全"):
            with st.spinner("补全中..."):
                # 将图像缩放到 256x256 用于模型输入
                resized_img = original.resize((256, 256), Image.LANCZOS)
                img_tensor = normalize_01_into_pm1(transforms.ToTensor()(resized_img)).unsqueeze(0).to(device)

                with torch.no_grad():
                    input_img_tokens = vae.img_to_idxBl(img_tensor, var.patch_nums)

                edit_mask = get_edit_mask(
                    var.patch_nums,
                    y0=y0, x0=x0,
                    y1=y1, x1=x1,
                    device=device,
                    inpainting=True
                )

                with torch.inference_mode():
                    with torch.autocast('cuda', enabled=True, dtype=torch.float16):
                        recon_B3HW = autoregressive_infer_cfg_with_mask(
                            var=var,
                            B=1,
                            label_B=torch.tensor([1000], device=device),
                            cfg=cfg_inp,
                            top_k=900,
                            top_p=0.96,
                            g_seed=0,
                            more_smooth=True,
                            input_img_tokens=input_img_tokens,
                            edit_mask=edit_mask,
                        )

                result_img = recon_B3HW[0].clamp(0,1).cpu().permute(1,2,0).numpy()
                result_img = (result_img * 255).astype(np.uint8)
                st.image(result_img, caption="补全结果", use_column_width=True)
st.sidebar.markdown("""
### 说明
- **图像生成**：选择类别，调整 CFG 强度，生成对应类别的图像。
- **局部补全**：上传图像，用滑块选择矩形区域，模型将自动补全该区域的内容（零样本编辑）。
""")