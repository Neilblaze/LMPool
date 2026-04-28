"""
FP8 native tensor casting example.

Leverages PyTorch's native 8-bit floating point representations (e4m3fn) 
to compress tensors while maintaining dynamic range.
Requires: PyTorch >= 2.1
"""
import torch

def main():
    print("Executing native FP8 tensor casting...")
    
    fp32_tensor = torch.randn(4, 4, dtype=torch.float32)
    print("Baseline FP32 tensor generated.")
    
    try:
        # e4m3fn format: 4 exponent bits, 3 mantissa bits (standard for inference weights)
        fp8_tensor = fp32_tensor.to(torch.float8_e4m3fn)
        print(f"Successfully cast to: {fp8_tensor.dtype}")
        
        # Cast back to measure the quantization penalty
        reconstructed = fp8_tensor.to(torch.float32)
        mae = torch.max(torch.abs(fp32_tensor - reconstructed)).item()
        
        print(f"Maximum absolute error post-reconstruction: {mae:.6f}")
        
    except AttributeError:
        print("Native FP8 requires PyTorch >= 2.1. Please upgrade your environment.")

if __name__ == "__main__":
    main()
