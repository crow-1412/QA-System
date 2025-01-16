from sentence_transformers import SentenceTransformer
import os
import torch
from tqdm import tqdm
import time

MODEL_PATHS = {
    "m3e": "/root/autodl-tmp/pre_train_model/m3e-large",
    "bge": "/root/autodl-tmp/pre_train_model/bge-large-zh-v1.5",
    "gte": "/root/autodl-tmp/pre_train_model/gte-large-zh",
    "bce": "/root/autodl-tmp/pre_train_model/bce-embedding-base_v1"
}

def download_and_check_models():
    """下载并检查所有需要的预训练模型"""
    print("\nStarting model initialization process...")
    
    for name, path in tqdm(MODEL_PATHS.items(), desc="Processing models"):
        try:
            print(f"\n{'='*50}")
            print(f"Processing {name} model...")
            
            if not os.path.exists(path):
                print(f"Model path {path} not found")
                print(f"Downloading {name} model...")
                start_time = time.time()
                model = SentenceTransformer(path)
                print(f"Download completed in {time.time() - start_time:.2f} seconds")
                
                print(f"Saving model to {path}...")
                model.save(path)
                print(f"Model saved successfully")
            else:
                print(f"Loading existing model from {path}...")
                start_time = time.time()
                model = SentenceTransformer(path)
                print(f"Model loaded in {time.time() - start_time:.2f} seconds")
            
            print("Testing model...")
            start_time = time.time()
            test_text = "这是一个测试句子"
            embedding = model.encode([test_text])
            print(f"Test completed in {time.time() - start_time:.2f} seconds")
            print(f"Embedding shape: {embedding.shape}")
            
            print("Cleaning up...")
            del model
            torch.cuda.empty_cache()
            print(f"Finished processing {name} model")
            
        except Exception as e:
            print(f"Error with {name} model: {str(e)}")
            raise

    print("\nAll models processed successfully!")

if __name__ == "__main__":
    download_and_check_models() 