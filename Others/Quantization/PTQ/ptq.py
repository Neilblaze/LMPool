"""
Post-Training Quantization (PTQ) example.

Demonstrates dynamic quantization on a PyTorch linear layer,
reducing memory footprint by converting FP32 weights to INT8.
"""
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
        
    def forward(self, x):
        return self.fc(x)

def main():
    model = SimpleModel()
    model.eval()
    
    dummy_input = torch.randn(1, 128)
    
    size_fp32 = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"FP32 model size: {size_fp32} bytes")
    
    # Apply dynamic quantization to INT8 for Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    size_int8 = sum(p.numel() for p in quantized_model.parameters())
    print(f"INT8 model size (approx): {size_int8} bytes")
    
    with torch.no_grad():
        out = quantized_model(dummy_input)
        
    print(f"Inference complete. Output shape: {out.shape}")

if __name__ == "__main__":
    main()
