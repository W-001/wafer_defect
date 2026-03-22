import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image

# 1. 硬件探测：优先使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的计算设备: {device}")

# 2. 严格参考官方 README 加载本地 DINOv3
# 指向您刚刚 git clone 下来的 dinov3 本地目录
REPO_DIR = '/home/xiaofan/project/dinov3' 
# 指向您下载的官方权重文件
WEIGHTS_PATH = f'{REPO_DIR}/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth' 

print("正在通过本地仓库加载官方 DINOv3 模型...")
# 使用 'dinov3_vitl16'，官方 README 提供的预定义模型名称之一
model = torch.hub.load(
    repo_or_dir=REPO_DIR, 
    model='dinov3_vitl16', 
    source='local', 
    weights=WEIGHTS_PATH
)
model = model.to(device)
model.eval() # 核心操作：设置为评估模式，彻底冻结网络权重

# 3. 构建官方建议的预处理流水线
# 官方 DINOv3 使用 ImageNet 的均值和方差，且输入尺寸推荐是 patch size(16) 的倍数
transform = T.Compose([
    T.Resize((384, 384)), # 调整到一致的尺度
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def process_sem_image(image_path):
    """执行 SEM 图像预处理与 DINOv3 官方特征提取"""
    # A. 模拟单通道晶圆灰度图读取
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"无法读取图片: {image_path}")
        
    # B. ROI 固定区域裁剪 (移除底部刻度线，假设占 15%)
    h, w = img_gray.shape
    crop_h = int(h * 0.85)
    img_cropped = img_gray[:crop_h, :]
    
    # C. 单通道转 3 通道 (R=G=B)
    img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2RGB)
    
    # D. OpenCV numpy 数组转 PIL Image，然后过官方 Transform
    pil_img = Image.fromarray(img_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device) # 增加 Batch 维度
    
    # E. 提取特征 (在 no_grad 上下文下执行)
    with torch.no_grad():
        # DINOv3 的前向传播返回结果可以直接获取 cls token 作为全局特征
        # 对于 ViT-Small，这将是一个 384 维的高维特征向量
        features = model(input_tensor)
        
        # 将特征转移到 CPU 并转为 numpy，方便后续的自监督聚类 (K-Means/DBSCAN) 或二分类
        image_features = features.cpu().numpy()
        
    return image_features

if __name__ == "__main__":
    # 快速验证环境：用 NumPy 伪造一张 512x512 的虚拟 SEM 噪声图片
    test_img_path = "dummy_wafer_sem.png"
    dummy_img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    cv2.imwrite(test_img_path, dummy_img)
    
    # 执行特征提取流水线
    print("开始提取图像特征...")
    features = process_sem_image(test_img_path)
    print(f"✅ 成功提取 DINOv3 官方特征！特征向量维度 (Batch, Dims): {features.shape}")