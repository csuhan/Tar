import os
import torch
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from torchvision.transforms.functional import to_tensor
from diffusers import SanaPipeline

from tok.ta_tok import TextAlignedTokenizer


class SanaAutoEncoder:
    def __init__(self, sana_path, ta_tok_path, device):
        self.device = device
        self.pipe = self.load_sana(sana_path, device=device)
        self.visual_tokenizer = TextAlignedTokenizer.from_checkpoint(
            ta_tok_path, load_teacher=False, input_type='indices'
        ).to(device)
        # negtive prompts for CFG
        self.neg_embeds = torch.load(os.path.join(sana_path, 'negative_prompt_embeds.pth'))[None].to(device)
        self.neg_attn_mask = torch.ones(self.neg_embeds.shape[:2], dtype=torch.int16, device=device)
    
    def load_sana(self, path, device='cuda'):
        pipe = SanaPipeline.from_pretrained(path, torch_dtype=torch.float32)
        pipe.to(device)
        pipe.text_encoder.to(torch.bfloat16)
        pipe.transformer = pipe.transformer.to(torch.bfloat16)
        return pipe

    def decode_from_encoder_indices(self, indices, **kwargs):
        caption_embds = self.visual_tokenizer.decode_from_bottleneck(indices).to(torch.bfloat16)
        caption_embds_mask = torch.ones(caption_embds.shape[:2], dtype=torch.int16, device=caption_embds.device)
        image = self.pipe(
            prompt_embeds=caption_embds,
            prompt_attention_mask=caption_embds_mask, 
            negative_prompt_embeds=self.neg_embeds,
            negative_prompt_attention_mask=self.neg_attn_mask,
            negative_prompt=None,
            **kwargs)
        return image
    
    def __call__(self, image_path, **kwargs):
        image = Image.open(image_path).convert('RGB')
        image = to_tensor(image).unsqueeze(0).to(self.device)
        indices = self.visual_tokenizer(image)['bottleneck_rep']
        return self.decode_from_encoder_indices(indices, **kwargs)

if __name__ == '__main__':
    sana_path = snapshot_download("csuhan/Tar-SANA-600M-1024px")
    ta_tok_path = hf_hub_download("csuhan/TA-Tok", "ta_tok.pth")
    device = "cuda"
    pipe = SanaAutoEncoder(sana_path, ta_tok_path, device)

    image_path = 'asset/dog_cat.jpg'
    generator = torch.Generator(device='cuda').manual_seed(42)
    out_image = pipe(
        image_path,
        height=1024, width=1024,
        num_inference_steps=28,
        guidance_scale=1.0,
        generator=generator)[0]
    out_image[0].save("output.png")