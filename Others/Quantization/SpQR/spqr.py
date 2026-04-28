"""
Sparse-Quantized Representation (SpQR) example.

Isolates critical outlier weights into a high-precision sparse matrix 
while aggressively quantizing the remaining standard weights to low bit-widths.
"""
import torch
import torch.nn as nn

def apply_spqr(weight: torch.Tensor, threshold_sigma: float = 3.0):
    """
    Conceptually separates weight matrices into sparse outliers and dense normal distributions.
    """
    std = weight.std()
    mean = weight.mean()
    
    # Identify outliers beyond the standard deviation threshold
    outlier_mask = torch.abs(weight - mean) > (threshold_sigma * std)
    
    # Maintain outliers in original precision
    sparse_outliers = weight.clone()
    sparse_outliers[~outlier_mask] = 0.0
    
    # Process dense body for low-bit quantization
    dense_body = weight.clone()
    dense_body[outlier_mask] = mean
    
    # Simulate a dense INT4 mapping
    scale = dense_body.abs().max() / 7.0
    quantized_body = torch.round(dense_body / scale).clamp(-8, 7)
    dequantized_body = quantized_body * scale
    
    # Merge the representations
    reconstructed = dequantized_body + sparse_outliers
    return reconstructed, outlier_mask.sum().item()

def main():
    layer = nn.Linear(1024, 1024)
    total_params = layer.weight.numel()
    
    print(f"Analyzing parameter distribution for a [{layer.weight.shape[0]}x{layer.weight.shape[1]}] matrix...")
    
    reconstructed_weight, outlier_count = apply_spqr(layer.weight)
    outlier_ratio = (outlier_count / total_params) * 100
    
    print(f"Isolated {outlier_count} structural outliers ({outlier_ratio:.2f}% of total footprint).")
    
    mse = nn.functional.mse_loss(layer.weight, reconstructed_weight)
    print(f"Reconstruction MSE: {mse.item():.6f}")

if __name__ == "__main__":
    main()
