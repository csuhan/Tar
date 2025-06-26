import argparse
import json
import os
import re

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import autocast
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model
from eval.eval_dpg_bench import get_prompt_template, load_visual_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=str, default="../geneval/prompts/evaluation_metadata.jsonl")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument("--ar_path", type=str, default='/tmp/ar_dtok_lp_256px.pth')
    parser.add_argument("--encoder_path", type=str, default='/tmp/ta_tok.pth')
    parser.add_argument("--decoder_path", type=str, default='/tmp/vq_ds16_t2i.pt')
    parser.add_argument('--seq_len', type=int, default=729)
    parser.add_argument('--seq_scale', type=int, default=1)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--repeat', type=int, default=4)
    return parser.parse_args()


class GenEvalDataset(Dataset):
    def __init__(self, args, tokenizer):
        self.prompts = [x.strip() for x in open(args.prompts).readlines()]
        prompt_temp=get_prompt_template(args)
        self.all_input_ids = []
        for i, prompt in enumerate(self.prompts):
            question = prompt_temp.format(json.loads(prompt)['prompt'])
            input_ids = tokenizer_image_token(question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")[None]
            self.all_input_ids.extend(input_ids)

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.all_input_ids[idx], self.prompts[idx], idx


if __name__ == '__main__':
    # distribtued init
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    args = parse_args()
    # load visual tokenizer
    args.ar_path = hf_hub_download("csuhan/TA-Tok", "ar_dtok_lp_256px.pth")
    args.encoder_path = hf_hub_download("csuhan/TA-Tok", "ta_tok.pth")
    args.decoder_path = hf_hub_download("peizesun/llamagen_t2i", "vq_ds16_t2i.pt")
    visual_tokenizer = load_visual_tokenizer(args).to(device)

    tokenizer, model, _, _ = load_pretrained_model(args.model, None, 'llava_qwen', device_map=device, multimodal=True)
    model.eval().to(device=device, dtype=dtype)

    dataset = GenEvalDataset(args, tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    seq_len = args.seq_len
    os.makedirs(args.save_dir, exist_ok=True)
    for i, data in enumerate(tqdm(dataloader)):
        input_ids, meta_data, idx = data
        input_ids = torch.cat([input_ids] * args.repeat)
        with autocast(dtype=model.dtype):
            cont = model.generate(
                input_ids.to(device), images=None,
                do_sample=True, temperature=1.0,
                max_new_tokens=args.seq_len)
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)
        
        codes = []
        for text_output in text_outputs:
                code = re.findall(r'<I(\d+)>', text_output)
                code = [int(x) for x in code]
                if len(code) < seq_len:
                    code = code + [0] * (seq_len - len(code))
                else:
                    code = code[:seq_len]
                codes.append(code)
        codes=torch.tensor(codes)[:, :seq_len].to(device)
        
        with torch.no_grad():
            recs = visual_tokenizer.decode_from_encoder_indices(codes, {'cfg_scale': 4.0})
        
        idx = int(idx)
        for j, rec in enumerate(recs):
            save_path = os.path.join(args.save_dir, f"{str(idx).zfill(5)}/samples/{str(j).zfill(5)}.png")
            save_sample_dir = os.path.dirname(save_path)
            os.makedirs(save_sample_dir, exist_ok=True)
            Image.fromarray(rec.numpy()).save(save_path)
        meta_save_path = os.path.join(args.save_dir, f"{str(idx).zfill(5)}/metadata.jsonl")
        with open(meta_save_path, 'w') as f:
            f.write(meta_data[0])
    
    dist.barrier()
    dist.destroy_process_group()