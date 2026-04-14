import torch
from transformers import pipeline, GenerationConfig

def run_translategemma_demo():
    model_id = "google/translategemma-4b-it"
    
    print(f"Loading {model_id}...")
    
    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        device_map="auto",
        dtype=torch.bfloat16
    )


    gen_config = GenerationConfig.from_pretrained(model_id)
    gen_config.max_length = None 
    gen_config.pad_token_id = pipe.tokenizer.eos_token_id

    # --- EXAMPLE 1: TEXT TRANSLATION---
    text_to_translate = "The translation of this sentence was done by a multimodal model."
    msg_text = [{
        "role": "user",
        "content": [{
            "type": "text",
            "source_lang_code": "en",
            "target_lang_code": "ca",
            "text": text_to_translate,
        }]
    }]

    print("\n--- Translating Text ---")
    gen_config.max_new_tokens = 200
    
    out_text = pipe(text=msg_text, generate_kwargs={"generation_config": gen_config})
    print(f"Translation: {out_text[0]['generated_text'][-1]['content']}")

    # --- EXAMPLE 2: IMAGE TRANSLATION ---
    image_url = "https://c7.alamy.com/comp/2YAX36N/traffic-signs-in-czech-republic-pedestrian-zone-2YAX36N.jpg"
    msg_image = [{
        "role": "user",
        "content": [{
            "type": "image",
            "source_lang_code": "cs",
            "target_lang_code": "ca",
            "url": image_url,
        }]
    }]

    print("\n--- Translating Image ---")
    out_image = pipe(text=msg_image, generate_kwargs={"generation_config": gen_config})
    print(f"Result: {out_image[0]['generated_text'][-1]['content']}")

if __name__ == "__main__":
    run_translategemma_demo()
