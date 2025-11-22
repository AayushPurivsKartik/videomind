# utils/vlm_captioner.py  ← PATCHED: Uses no-flash-attn fork
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
import os

class Florence2Captioner:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            print("Loading Florence-2 large (no-flash-attn patched version)... (~30-60 sec first time)")
            cls._instance = super(Florence2Captioner, cls).__new__(cls)
            
            # Use the community fork that removes flash_attn dependency
            model_id = "multimodalart/Florence-2-large-no-flash-attn"
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            cls._instance.model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True
            ).to(cls._instance.device).eval()
            
            cls._instance.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            print(f"Florence-2 loaded on {cls._instance.device}! (No flash_attn needed)")
        return cls._instance

    @torch.no_grad()
    def describe(self, image_path: str, detail: str = "detailed") -> str:
        image = Image.open(image_path).convert("RGB")
        
        if detail == "detailed":
            task_prompt = "<DETAILED_CAPTION>"
        elif detail == "short":
            task_prompt = "<CAPTION>"
        else:
            task_prompt = "<DETAILED_CAPTION>"
        
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=128,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()

# Global singleton
captioner = Florence2Captioner()