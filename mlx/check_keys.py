import mlx.core as mx
import sys
import glob

# 모델 경로 지정
model_path = "./models/deepseek-moe-16b-chat-mlx-q4_0"  # 실제 경로로 수정

# safetensors 파일 찾기
files = glob.glob(f"{model_path}/*.safetensors")
if not files:
    print(f"Error: No safetensors found in {model_path}")
    sys.exit(1)

print(f"Loading metadata from {files[0]}...")
weights = mx.load(files[0])

print(f"\nTotal keys found: {len(weights)}")
print("=== First 10 Keys ===")
for i, key in enumerate(weights.keys()):
    if i >= 10: break
    print(f"Key: {key} | Shape: {weights[key].shape}")

print("\n=== Checking Essential Keys ===")
essentials = ["embed_tokens", "tok_embeddings", "norm", "lm_head", "output"]
for term in essentials:
    found = [k for k in weights.keys() if term in k]
    if found:
        print(f"Found '{term}': {found[0]}")
    else:
        print(f"Missing '{term}'")

