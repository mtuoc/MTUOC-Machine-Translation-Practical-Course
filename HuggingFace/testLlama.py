import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer

def run_llama_inference():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config
    )

    prompt_text = "Explain the importance of open-source AI models in three bullet points."

    messages = [
        {"role": "system", "content": "You are a technical assistant."},
        {"role": "user", "content": prompt_text}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    # Initialize the streamer
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    print(f"Generating response for model: {model_id}\n")
    print("-" * 30)

    # The generate method will now print tokens as they are created
    model.generate(
        **inputs,
        max_new_tokens=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )
    
    print("-" * 30)

if __name__ == "__main__":
    run_llama_inference()
