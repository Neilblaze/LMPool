"""
QLoRA (Quantized Low-Rank Adaptation) example.

Injects trainable low-rank adapters into a frozen, 4-bit quantized base model,
enabling highly parameter-efficient fine-tuning without catastrophic forgetting.
Requires: pip install transformers peft bitsandbytes
"""
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

def main():
    model_id = "facebook/opt-125m"
    print(f"Configuring QLoRA training pipeline for {model_id}...")
    
    try:
        import torch
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        print("Base model quantization and LoRA adapter constraints established.")
        print("Note: Uncomment the execution block to inject adapters and inspect parameters.")
        
        # from transformers import AutoModelForCausalLM
        # from peft import prepare_model_for_kbit_training, get_peft_model
        #
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_id, quantization_config=bnb_config, device_map="auto"
        # )
        # model = prepare_model_for_kbit_training(model)
        # peft_model = get_peft_model(model, lora_config)
        # peft_model.print_trainable_parameters()
        
    except ImportError:
        print("Missing dependencies. Install via: pip install transformers peft bitsandbytes")

if __name__ == "__main__":
    main()
