import re
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoTokenizer, Qwen2ForCausalLM

from tok.mm_autoencoder import MMAutoEncoder


@dataclass
class T2IConfig:
    model_path: str = "csuhan/Tar-1.5B"
    # visual tokenizer config
    ar_path: str = 'ar_dtok_lp_256px.pth'
    encoder_path: str = 'ta_tok.pth'
    decoder_path: str = 'vq_ds16_t2i.pt'
    
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    # generation parameters
    scale: int = 0  # choose from [0, 1, 2]
    seq_len: int = 729  # choose from [729, 169, 81]
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 1200
    cfg_scale: float = 4.0

class TextToImageInference:
    def __init__(self, config: T2IConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._load_models()
        
    def _load_models(self):
        self.model = Qwen2ForCausalLM.from_pretrained(self.config.model_path, torch_dtype=self.config.dtype).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        
        # Initialize visual tokenizer
        config = dict(
            ar_path=self.config.ar_path,
            encoder_path=self.config.encoder_path,
            decoder_path=self.config.decoder_path,
            encoder_args={'input_type': 'rec'},
            decoder_args={},
        )
        self.visual_tokenizer = MMAutoEncoder(**config).eval().to(dtype=self.config.dtype, device=self.device)
        self.visual_tokenizer.ar_model.cls_token_num = self.config.seq_len
        self.visual_tokenizer.encoder.pool_scale = self.config.scale + 1

    def generate_image(self, prompt: str) -> Image.Image:
        # Prepare prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        input_text += f"<im_start><S{self.config.scale}>"
        
        # Generate tokens
        inputs = self.tokenizer(input_text, return_tensors="pt")
        gen_ids = self.model.generate(
            inputs.input_ids.to(self.device),
            max_new_tokens=self.config.seq_len,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k)
        
        # Process generated tokens
        gen_text = self.tokenizer.batch_decode(gen_ids)[0]
        gen_code = [int(x) for x in re.findall(r'<I(\d+)>', gen_text)]
        gen_code = gen_code[:self.config.seq_len] + [0] * max(0, self.config.seq_len - len(gen_code))
        gen_code = torch.tensor(gen_code).unsqueeze(0).to(self.device)
        
        gen_tensor = self.visual_tokenizer.decode_from_encoder_indices(
            gen_code, 
            {'cfg_scale': self.config.cfg_scale}
        )
        gen_image = Image.fromarray(gen_tensor[0].numpy())
        return gen_image

def main():
    config = T2IConfig()
    config.ar_path = hf_hub_download("csuhan/TA-Tok", "ar_dtok_lp_1024px.pth")
    config.encoder_path = hf_hub_download("csuhan/TA-Tok", "ta_tok.pth")
    config.decoder_path = hf_hub_download("peizesun/llamagen_t2i", "vq_ds16_t2i.pt")
    inference = TextToImageInference(config)
    
    prompt = "A photo of a macaw"
    image = inference.generate_image(prompt)
    image.save("generated_image.png")

if __name__ == "__main__":
    main()