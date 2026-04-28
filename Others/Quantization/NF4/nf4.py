"""
NormalFloat4 (NF4) example.

An information-theoretically optimal data type specifically engineered for 
normally distributed neural network weights. Foundational to QLoRA.
Requires: pip install transformers bitsandbytes accelerate
"""
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

def main():
    model_id = "facebook/opt-125m"
    print(f"Generating NF4 configuration for {model_id}...")
    
    try:
        import torch
        from transformers import BitsAndBytesConfig
        
        # Double quantization provides further memory savings by quantizing the quantization constants
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        print("NF4 configuration initialized successfully.")
        print("Note: Uncomment the model loading pipeline to initiate the 4-bit conversion.")
        
        # from transformers import AutoModelForCausalLM
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_id,
        #     quantization_config=bnb_config,
        #     device_map="auto"
        # )
        # print("Model weights successfully cast to NormalFloat4.")
        
    except ImportError:
        print("Missing dependencies. Install via: pip install transformers bitsandbytes accelerate")

if __name__ == "__main__":
    main()
