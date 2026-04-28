"""
Quantization-Aware Training (QAT) example.

Simulates the effects of quantization during the forward and backward 
passes to minimize accuracy degradation before final conversion to INT8.
"""
import torch
import torch.nn as nn
import torch.optim as optim

class QATModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.fc = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dequant(x)
        return x

def main():
    model = QATModel()
    model.train()
    
    # Configure quantization for x86 via FBGEMM
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("Initiating QAT forward/backward passes...")
    for epoch in range(3):
        optimizer.zero_grad()
        inputs = torch.randn(10, 32)
        targets = torch.randn(10, 16)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1} | Loss: {loss.item():.4f}")
        
    model.eval()
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    test_input = torch.randn(1, 32)
    with torch.no_grad():
        out = quantized_model(test_input)
        
    print(f"QAT conversion successful. Final tensor shape: {out.shape}")

if __name__ == "__main__":
    main()
