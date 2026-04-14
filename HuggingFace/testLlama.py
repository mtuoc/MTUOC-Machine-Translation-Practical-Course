import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_llama_inference():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16
    )

    prompt_text = "Explain the importance of open-source AI models in three bullet points."

    messages = [
        {"role": "system", "content": "You are a technical assistant."},
        {"role": "user", "content": prompt_text}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    attention_mask = torch.ones(input_ids.shape, device=model.device)

    print(f"Generating response for model: {model_id}")

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    print("-" * 30)
    print(f"Prompt: {prompt_text}")
    print("-" * 30)
    print(f"Response: {response}")
    print("-" * 30)

if __name__ == "__main__":
    run_llama_inference()
