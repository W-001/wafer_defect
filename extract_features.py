import torch
from transformers import AutoImageProcessor, AutoModel
import cv2
import numpy as np

# 1. 硬件探测：优先使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的计算设备: {device}")

# 2. 加载 DINOv3 冻结骨干网络 (使用 ViT-Base 版本，平衡特征表征力与推理速度)
# 首次运行会自动从 Hugging Face 拉取预训练权重
model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
print("正在加载 DINOv3 模型与预处理器...")
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
model.eval() # 核心操作：设置为评估模式，彻底冻结网络权重

def process_sem_image(image_path):
    """执行 SEM 图像预处理与特征提取"""
    # A. 模拟单通道晶圆灰度图读取
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"无法读取图片: {image_path}")
        
    # B. 架构图中的 Step 1: ROI 固定区域裁剪 (假设底部 15% 是刻度线)
    h, w = img_gray.shape
    crop_h = int(h * 0.85)
    img_cropped = img_gray[:crop_h, :]
    
    # C. 架构图中的 Step 3: 单通道转 3 通道 (R=G=B 复制)
    img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2RGB)
    
    # D. 将图像转换为 DINOv3 所需的 Tensor 格式
    inputs = processor(images=img_rgb, return_tensors="pt").to(device)
    
    # E. 提取特征 (在 no_grad 上下文下执行，节省显存，绝对不更新权重)
    with torch.no_grad():
        outputs = model(**inputs)
        # 提取 [CLS] token 的特征向量，它代表了整张图片的全局高维语义总结
        # 对于 ViT-Base，这将是一个 768 维的向量
        image_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
    return image_features

if __name__ == "__main__":
    # 为了快速验证环境，我们先用 NumPy 伪造一张 384x384 的虚拟 SEM 噪声图片
    test_img_path = "dummy_sem.png"
    dummy_img = np.random.randint(0, 255, (384, 384), dtype=np.uint8)
    cv2.imwrite(test_img_path, dummy_img)
    
    # 执行特征提取流水线
    print("开始提取图像特征...")
    features = process_sem_image(test_img_path)
    print(f"✅ 成功提取 DINOv3 特征！特征向量维度 (Batch, 维度): {features.shape}")