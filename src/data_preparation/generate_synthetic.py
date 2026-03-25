"""
Stable Diffusion 合成圖像生成器
用於生成訓練用的 AI 生成圖像
"""

import torch
from pathlib import Path
from typing import List, Optional, Dict
import json
from tqdm import tqdm

def generate_synthetic_images(
    output_dir: str,
    num_images: int = 100,
    model_id: str = "stabilityai/stable-diffusion-2-1",
    device: str = "cuda",
    batch_size: int = 4,
    seed: Optional[int] = 42
):
    """
    使用 Stable Diffusion 生成合成圖像
    
    Args:
        output_dir: 輸出目錄
        num_images: 生成圖像數量
        model_id: Stable Diffusion 模型 ID
        device: 運算裝置
        batch_size: 批次大小
        seed: 隨機種子
    """
    try:
        from diffusers import StableDiffusionPipeline
        
        print(f"🎨 開始生成合成圖像...")
        print(f"   模型: {model_id}")
        print(f"   數量: {num_images}")
        print(f"   裝置: {device}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 載入 Stable Diffusion 模型
        print("\n⏳ 載入 Stable Diffusion 模型...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None  # 關閉安全檢查器以加速
        )
        pipe = pipe.to(device)
        
        # 啟用記憶體優化
        if device == "cuda":
            pipe.enable_attention_slicing()
            print("✅ GPU 記憶體優化已啟用")
        
        print("✅ 模型載入完成")
        
        # 生成多樣化的 prompts
        prompts = generate_diverse_prompts(num_images)
        
        # 設置隨機種子
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None
        
        # 生成圖像
        metadata = []
        generated_count = 0
        
        print(f"\n🖼️  開始生成圖像...")
        for i in tqdm(range(0, num_images, batch_size), desc="Generating"):
            batch_prompts = prompts[i:i+batch_size]
            
            # 生成批次圖像
            images = pipe(
                batch_prompts,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=generator,
                height=512,
                width=512
            ).images
            
            # 儲存圖像
            for j, (image, prompt) in enumerate(zip(images, batch_prompts)):
                img_idx = i + j
                img_path = output_dir / f"synthetic_{img_idx:06d}.png"
                image.save(img_path)
                
                # 記錄元資料
                metadata.append({
                    'filename': img_path.name,
                    'prompt': prompt,
                    'model': model_id,
                    'seed': seed,
                    'index': img_idx
                })
                
                generated_count += 1
        
        # 儲存元資料
        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 合成圖像生成完成！")
        print(f"   生成數量: {generated_count}")
        print(f"   儲存路徑: {output_dir}")
        print(f"   元資料: {metadata_path}")
        
        # 清理 GPU 記憶體
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return {
            'generated_count': generated_count,
            'output_dir': str(output_dir),
            'metadata_path': str(metadata_path)
        }
        
    except Exception as e:
        print(f"❌ 生成圖像時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_diverse_prompts(num_prompts: int) -> List[str]:
    """
    生成多樣化的圖像描述 prompts
    
    Args:
        num_prompts: 需要生成的 prompt 數量
        
    Returns:
        prompts 列表
    """
    # Prompt 模板（覆蓋各種場景）
    templates = {
        'portrait': [
            "a photo of a person, professional photography",
            "portrait of a young woman, natural lighting",
            "headshot of a businessman, studio lighting",
            "close-up portrait of an elderly man, warm colors",
            "selfie of a teenager, casual style",
            "professional portrait, corporate style",
            "artistic portrait, dramatic lighting",
            "candid portrait, natural expression"
        ],
        'landscape': [
            "beautiful mountain landscape, sunset",
            "serene beach scene, blue sky",
            "forest path, autumn colors",
            "city skyline at night, urban photography",
            "countryside view, green fields",
            "desert landscape, golden hour",
            "tropical beach, crystal clear water",
            "snowy mountains, winter scene"
        ],
        'object': [
            "a red apple on a wooden table",
            "modern smartphone, product photography",
            "vintage camera, detailed shot",
            "coffee cup on white background",
            "sports car, dynamic angle",
            "flower bouquet, macro photography",
            "laptop computer, clean background",
            "wristwatch, luxury style"
        ],
        'scene': [
            "busy city street, daytime",
            "cozy living room, modern interior",
            "restaurant interior, warm ambiance",
            "park with people, sunny day",
            "office workspace, professional",
            "library interior, peaceful atmosphere",
            "shopping mall, crowded scene",
            "train station, urban life"
        ],
        'animal': [
            "cute cat, fluffy fur",
            "golden retriever dog, happy expression",
            "colorful parrot, tropical bird",
            "wild lion, majestic pose",
            "swimming dolphin, ocean scene",
            "butterfly on flower, macro shot",
            "horse in field, pastoral scene",
            "elephant in savanna, wildlife photography"
        ]
    }
    
    prompts = []
    categories = list(templates.keys())
    
    for i in range(num_prompts):
        # 輪流選擇類別
        category = categories[i % len(categories)]
        # 選擇該類別的 prompt
        category_prompts = templates[category]
        prompt = category_prompts[i % len(category_prompts)]
        prompts.append(prompt)
    
    return prompts

def test_generation(output_dir: str = "test_output", num_images: int = 5):
    """
    測試生成功能（少量圖像）
    
    Args:
        output_dir: 測試輸出目錄
        num_images: 測試圖像數量
    """
    print("🧪 開始測試合成圖像生成...")
    result = generate_synthetic_images(
        output_dir=output_dir,
        num_images=num_images,
        batch_size=1
    )
    
    if result:
        print("\n✅ 測試成功！")
        print(f"   請檢查: {result['output_dir']}")
    else:
        print("\n❌ 測試失敗")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic images using Stable Diffusion')
    parser.add_argument('--output_dir', type=str, default='data/synthetic',
                       help='Output directory')
    parser.add_argument('--num_images', type=int, default=100,
                       help='Number of images to generate')
    parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1',
                       help='Stable Diffusion model ID')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for generation')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--test', action='store_true',
                       help='Run test with 5 images')
    
    args = parser.parse_args()
    
    if args.test:
        test_generation()
    else:
        generate_synthetic_images(
            output_dir=args.output_dir,
            num_images=args.num_images,
            model_id=args.model_id,
            device=args.device,
            batch_size=args.batch_size
        )
