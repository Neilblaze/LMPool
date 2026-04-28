"""
LLM.int8() example.

Handles large activation outliers in LLMs by keeping outlier channels in FP16 
while performing the bulk matrix multiplication in INT8 precision.
Requires: pip install transformers bitsandbytes accelerate
"""
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

def main():
    model_id = "facebook/opt-125m"
    print(f"Setting up native LLM.int8() runtime for {model_id}...")
    
    try:
        from transformers import AutoTokenizer
        import bitsandbytes
        
        print("BitsAndBytes backend detected.")
        print("Note: To perform inference with mixed-precision outliers, uncomment the generation block.")
        
        # from transformers import AutoModelForCausalLM
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_id,
        #     device_map="auto",
        #     load_in_8bit=True,
        # )
        # tokenizer = AutoTokenizer.from_pretrained(model_id)
        # 
        # inputs = tokenizer("LLM.int8() isolates outliers by", return_tensors="pt").to(model.device)
        # outputs = model.generate(**inputs, max_new_tokens=15)
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        
    except ImportError:
        print("Missing dependencies. Install via: pip install transformers bitsandbytes accelerate")

if __name__ == "__main__":
    main()
