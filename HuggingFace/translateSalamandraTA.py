import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def translate_salamandra_manual(text, model_id, source_lang, target_lang):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16
    )

    # 1. Construcció manual del prompt seguint el format de la fitxa
    # Definim un System Prompt per donar context al model
    system_prompt = "You are a professional translator."
    user_prompt = f"Translate the following text from {source_lang} into {target_lang}.\n{source_lang}: {text} \n{target_lang}:"

    # Ajuntem les peces amb els delimitadors <|im_start|> i <|im_end|>
    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # 2. Tokenització
    inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_length = inputs.input_ids.shape[1]

    # 3. Generació
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=400,
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>") # Forcem que sàpiga quan parar
    )

    # 4. Decodificació
    result = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    # Netegem possibles restes del delimitador final si n'hi hagués
    return result.replace("<|im_end|>", "").strip()

if __name__ == "__main__":
    model_id = "BSC-LT/salamandraTA-2b-instruct"
    
    # Exemple d'ús
    src = "English"
    tgt = "Catalan"
    frase = "Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making.."

    traduccio = translate_salamandra_manual(frase, model_id, src, tgt)
    print(f"\nTraducció: {traduccio}")
