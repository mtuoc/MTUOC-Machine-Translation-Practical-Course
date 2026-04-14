import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def translate_tower(text, model_id, source_lang, target_lang):
    """
    Tradueix text utilitzant Tower-Plus-2B (Unbabel).
    """
    # 1. Carregar tokenizer i model (usant 'dtype' en lloc de 'torch_dtype')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Tower sol necessitar que definim el pad_token si no ho està
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16
    )

    # 2. Preparar el missatge segons el format de Tower
    # Tower és molt sensible a l'idioma especificat entre parèntesis si s'escau
    messages = [
        {
            "role": "user", 
            "content": f"Translate the following {source_lang} source text to {target_lang}:\n{source_lang}: {text}\n{target_lang}: "
        }
    ]

    # Apliquem la plantilla oficial del model
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # 3. Tokenitzar i generar la attention_mask per evitar el warning
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_length = inputs.input_ids.shape[1]

    # Generació (Tower acostuma a funcionar bé amb do_sample=False o num_beams petit)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=256,
        do_sample=False,  # Com diu l'exemple d'Unbabel
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # 4. Decodificar i netejar
    # Amb Tower, el xat template a vegades retorna tot el diàleg
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Per obtenir només la traducció, tallem pel prompt d'entrada
    # En els models instruct, sovint és més net fer-ho així:
    generated_tokens = outputs[0][input_length:]
    translation = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return translation.strip()

if __name__ == "__main__":
    model_id = "Unbabel/Tower-Plus-2B"
    
    # Tower-Plus és molt bo amb llengües romàniques
    src = "English"
    tgt = "Portuguese (Portugal)"  # Pots usar "Catalan", "Spanish", etc.
    sentence = "The bridge between two languages is built with artificial intelligence."

    print(f"Carregant {model_id}...")
    resultat = translate_tower(sentence, model_id, src, tgt)

    print("-" * 30)
    print(f"Source: {sentence}")
    print(f"Target: {resultat}")
    print("-" * 30)
