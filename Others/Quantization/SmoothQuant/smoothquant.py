"""
SmoothQuant example.

Migrates quantization difficulty from activations to weights mathematically,
enabling robust W8A8 (Weight 8-bit, Activation 8-bit) inference for LLMs.
"""
import torch
import torch.nn as nn

def apply_smoothquant(layer: nn.Linear, alpha: float = 0.5):
    """
    Simulates the activation/weight smoothing process.
    """
    with torch.no_grad():
        weight = layer.weight
        
        # Simulate maximum activation magnitude per channel (usually profiled from a calibration set)
        activation_max = torch.rand(weight.shape[1]) * 10 
        weight_max = torch.max(torch.abs(weight), dim=0)[0]
        
        # Compute smoothing scales: s = max(|X|)^alpha / max(|W|)^(1-alpha)
        scales = (activation_max ** alpha) / (weight_max ** (1 - alpha) + 1e-5)
        
        # Scale the weights directly. In practice, activations are inversely scaled 
        # prior to the matrix multiplication to maintain mathematical equivalence.
        smoothed_weight = weight * scales.unsqueeze(0)
        layer.weight.data = smoothed_weight
        
        return layer, scales

def main():
    layer = nn.Linear(64, 128)
    original_std = layer.weight.std().item()
    
    smoothed_layer, scales = apply_smoothquant(layer, alpha=0.5)
    
    print(f"Original weight standard deviation: {original_std:.4f}")
    print(f"Smoothed weight standard deviation: {smoothed_layer.weight.std().item():.4f}")
    print(f"Mean smoothing scale applied: {scales.mean().item():.4f}")

if __name__ == "__main__":
    main()
