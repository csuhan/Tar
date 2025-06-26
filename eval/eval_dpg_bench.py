import argparse
import glob
import os
import re

import numpy as np
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
from tok.mm_autoencoder import MMAutoEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=str, default="../dpg_bench/prompts")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument("--ar_path", type=str, default='/tmp/ar_dtok_lp_256px.pth')
    parser.add_argument("--encoder_path", type=str, default='/tmp/ta_tok.pth')
    parser.add_argument("--decoder_path", type=str, default='/tmp/vq_ds16_t2i.pt')
    parser.add_argument('--seq_len', type=int, default=729)
    parser.add_argument('--seq_scale', type=int, default=1)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--repeat', type=int, default=4)
    return parser.parse_args()


def load_visual_tokenizer(args):
    config = dict(
        ar_path=args.ar_path,
        encoder_path=args.encoder_path,
        decoder_path=args.decoder_path,
        encoder_args={'input_type': 'rec'},
        decoder_args={})
    visual_tokenizer = MMAutoEncoder(**config).eval()
    visual_tokenizer.ar_model.cls_token_num = args.seq_len
    visual_tokenizer.encoder.pool_scale = args.seq_scale
    return visual_tokenizer


def get_prompt_template(args):
    return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<im_start>"+f"<S{str(args.seq_scale-1)}>"


class DPGBenchDataset(Dataset):
    def __init__(self, args, tokenizer):
        prompt_files = glob.glob(os.path.join(args.prompts, '*.txt'))
        self.prompt_names = [os.path.basename(x).split('.')[0] for x in prompt_files]
        self.prompts = [open(x).read().strip() for x in prompt_files]

        prompt_temp= get_prompt_template(args)
        self.all_input_ids = []
        for i, prompt in enumerate(self.prompts):
            question = prompt_temp.format(prompt)
            input_ids = tokenizer_image_token(question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")[None]            
            self.all_input_ids.extend(input_ids)
    
    def __len__(self):
        return len(self.all_input_ids)
    
    def __getitem__(self, idx):
        return self.all_input_ids[idx], self.prompt_names[idx]

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

    # load LLM
    tokenizer, model, _, _ = load_pretrained_model(args.model, None, 'llava_qwen', device_map=device, multimodal=True)
    model.eval().to(device=device, dtype=dtype)

    dataset = DPGBenchDataset(args, tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    seq_len = args.seq_len
    os.makedirs(args.save_dir, exist_ok=True)
    for i, data in enumerate(tqdm(dataloader)):
        input_ids, file_names = data
        input_ids = torch.cat([input_ids] * args.repeat)
        with autocast(dtype=model.dtype):
            cont = model.generate(
                input_ids.to(device), images=None,
                do_sample=True, temperature=1.0,
                max_new_tokens=seq_len)
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
            recs = recs.numpy()

        # cocat into a large image for eval
        top_row = np.concatenate((recs[0], recs[1]), axis=1)
        bottom_row = np.concatenate((recs[2], recs[3]), axis=1)
        final_image = np.concatenate((top_row, bottom_row), axis=0)
        save_path = os.path.join(args.save_dir, file_names[0]+'.png')
        Image.fromarray(final_image).save(save_path)
    
    dist.barrier()
    dist.destroy_process_group()