import megabyte
import torch
import torch.nn as nn
from einops import rearrange

import torchaudio
from ljspeech import LJSPEECH
DATASET_PATH = "./data/LJSpeech/"

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split

from einops import rearrange
from tqdm import tqdm
from encodec_util import decode_to_file

import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler

import math
import random
import numpy as np

from g2p_util import _get_model

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

MAX_LR         = 1e-3 # 1e-3
MIN_LR         = 1e-4
# MAX_LR         = 1e-2
WEIGHT_DECAY   = 1e-4
GRAD_CLIP      = 0.1
WARMUP_ITERS   = 500 # 500   # Taken from from MegaByte paper
EPOCHS         = 100 # 100 # 200 # 100 # 1000
PRINT_INTERVAL = 1
SEQ_LEN        = 1024 # 512
BATCH_SIZE     = 48 # 32 # 96 # 128 # 64
DECAY_LR       = True
NUM_BATCHES    = None # EPOCHS * # int(1e5)

BANDWIDTH_IDX     = 0 # original VALL-E
CODEBOOKS         = [2, 4, 8, 16, 32]
BANDWIDTHS        = [1.5, 3.0, 6.0, 12.0, 24.0]
BANDWIDTH         = BANDWIDTHS[BANDWIDTH_IDX]
CODEBOOK          = CODEBOOKS[BANDWIDTH_IDX]
MAX_CLIP_LENGTH   = int(5)
MAX_PROMPT_LENGTH = int(30 * MAX_CLIP_LENGTH)

VALIDATE_EVERY    = 1

AMP               = True
SAVE              = True

def get_reserved_mem_gb():
    device = torch.cuda.current_device()
    reserved = torch.cuda.memory_reserved(device)
    reserved_gb = reserved / 1024 / 1024 / 1024
    return reserved_gb

# Taken from: https://github.com/karpathy/nanoGPT/blob/master/train.py
# Learning rate decay scheduler (Cosine with Warmup)
def get_lr(it):
    # 1) Linear warmup for warmup_iters steps
    if it < WARMUP_ITERS:
        return MAX_LR * it / WARMUP_ITERS
    
    # 2) If it > lr_decay_iters, return min learning rate
    if it > NUM_BATCHES:
        return MIN_LR
    
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - WARMUP_ITERS) / (NUM_BATCHES - WARMUP_ITERS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return MIN_LR + coeff * (MAX_LR - MIN_LR)

class PhoneLM(nn.Module):
    def __init__(self, n_phone_tokens, n_audio_tokens):
        super(PhoneLM, self).__init__()
        self.megabyte   = megabyte.MEGABYTE(
            heads       = 16, # 1,
            dim_head    = 64, # 16,
            num_tokens  = n_phone_tokens + n_audio_tokens + 4,
            dim         = (1024, 256, 128), # (32, 32, 32), # (768, 256, 128)# Dg, Dl1, Dl2
            depth       = (6, 4, 2), # (12, 4, 2), # (6, 4, 2)
            max_seq_len = (SEQ_LEN // 16, 4, 4), # (32, 4, 4), # (128, 4, 4), # (32, 4, 4), # 512
            flash_attn  = False)

    def forward(self, x, debug=False, return_loss=True):
        x = self.megabyte(x, return_loss=return_loss)
        return x
    
    def get_params(self):
        o = [param.numel() for param in self.parameters() if param.requires_grad]
        o = sum(o)
        return o
    
    def generate(self, *args):
        return self.megabyte.generate(*args)
    
def multi_encode(
        phone_tokens,
        audio_tokens,
        n_phone_tokens,
        n_audio_tokens,
        max_clip_length=1.0):
    """NOTE: 75 steps per second for 24kHz in `encodec.
    Set `max_clip_length` to 0 for original clip length."""

    # Start text token, end text token, start audio token, end audio token
    ETT, EAT = [n_phone_tokens + n_audio_tokens + i
                for i in range(2)]
    ETT = torch.tensor([ETT]).long().cuda()
    EAT = torch.tensor([EAT]).long().cuda()

    if max_clip_length > 0:
        #print("pre audio_tokens.shape", audio_tokens.shape)
        audio_tokens = audio_tokens[:, :, :int(max_clip_length * 75)]
    #print("post audio_tokens.shape", audio_tokens.shape)
    audio_tokens = rearrange(audio_tokens.squeeze(0), "q s -> (q s)")
    #print("post einops audio_tokens.shape", audio_tokens.shape)
    
    # offset phone tokens past audio tokens
    phone_tokens += n_audio_tokens
    
    #print("phone_tokens.shape:", phone_tokens.shape)
    #print("audio_tokens.shape:", audio_tokens.shape)
    
    device = torch.cuda.current_device()
    phone_tokens = phone_tokens.to(device)
    # phone_tokens = torch.cat((phone_tokens), dim=0).to(device)
    audio_tokens = torch.cat((audio_tokens, EAT), dim=0).to(device)
    combined_tokens = torch.cat((phone_tokens, ETT, audio_tokens), dim=0).to(device)
    return phone_tokens, audio_tokens, combined_tokens

def generate_audio(sample,
                   n_phone_tokens,
                   n_audio_tokens,
                   audio_path="./out.wav"):
    ETT, EAT = [n_phone_tokens + n_audio_tokens + i
                          for i in range(2)]
    ST_S = [ETT, EAT]
    print("ETT, EAT ids:", ST_S)
    seq = sample.cpu().tolist()[0]
    print("seq:", seq)
    # all special tokens in list
    if all(st_t in seq for st_t in ST_S) and len(seq) >= len(ST_S) + 2:
        # text_tokens  = seq[seq.index(STT + 1):seq.index(ETT - 1)]
        audio_tokens = seq[seq.index(ETT)+1:seq.index(EAT)]
        print(seq.index(ETT), seq.index(EAT), len(audio_tokens))
        audio_tokens = torch.tensor(audio_tokens).cuda()
        audio_tokens = rearrange(
            audio_tokens,
            '(t q) -> t q',
            q=1, # CODEBOOK,
            t=audio_tokens.size(0) // 1) # t=audio_tokens.size(0) // CODEBOOK)
        print("audio_tokens.shape:", audio_tokens, audio_tokens.shape)
        decode_to_file(audio_tokens, audio_path)
        return True
    else:
        return False

def generate(model, prompt):
    model.eval()

    prompt = prompt.unsqueeze(0)
    sample = model.generate(prompt)
    sample = sample.flatten(1)
    # print("sample:", sample, sample.shape)

    return prompt, sample

def collate_fn(dataset, batch):
    """
    batch := [
        fileid_audio,
        waveform,
        sample_rate,
        transcript,
        normalized_transcript,
        phones,
        phone_ids,
        codes
    ] * batch_size
    """
    # print("collate len batch:", len(batch))
    items = []
    for item in batch:
        _, _, test_inp = multi_encode(
            item[-2],
            item[-1],
            n_phone_tokens=len(dataset.phone_dict),
            n_audio_tokens=1024,
            max_clip_length=MAX_CLIP_LENGTH)
        
        padding_len = max(0, SEQ_LEN - test_inp.size(0))
        n_test_inp = F.pad(test_inp, (0, padding_len))
        # print("n_test_inp.shape:", n_test_inp.shape)
        items.append(n_test_inp)

    out = torch.stack(items).cuda()
    # print(out.shape)
    return out
    # nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)

if __name__ == "__main__":
    dataset = LJSPEECH("./data/LJSpeech",
                       encodec_bandwidth=BANDWIDTH,
                       max_prompt_length=MAX_PROMPT_LENGTH)
    print("LJSpeech Dataset Slice:", len(dataset))

    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler  = SubsetRandomSampler(test_indices)
    eval_sampler  = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=lambda batch: collate_fn(dataset, batch))
    test_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        collate_fn=lambda batch: collate_fn(dataset, batch))
    eval_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=test_sampler,
        collate_fn=lambda batch: batch)

    print("len(train_loader), len(test_loader):", len(train_loader), len(test_loader))
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhoneLM(
        n_phone_tokens=len(dataset.phone_dict),
        n_audio_tokens=1024).to(device)

    best_val = float("inf")

    NUM_BATCHES = EPOCHS * int(math.ceil(len(dataset) / BATCH_SIZE))
    print("NUM_BATCHES:", NUM_BATCHES)

    # print("Model params:", model.megabyte.get_num_params())

    # item = next(iter(train_loader))[1]
    # item_phone_tokens = item[-2]
    # item_audio_tokens = item[-1]
    # item_phone_tokens.shape, item_audio_tokens.shape
    # print(item[0], item[3])

    # print("item_audio_tokens:", item_audio_tokens)
    # print("item_phone_tokens:",
    #       item_phone_tokens,
    #       [_get_model().phonemes[ph_id]
    #        for ph_id in item_phone_tokens
    #        if ph_id < len(_get_model().phonemes)])

    # phone_prompt, audio_target, test_inp = multi_encode(
    #     item_phone_tokens,
    #     item_audio_tokens,
    #     n_phone_tokens=len(dataset.phone_dict),
    #     n_audio_tokens=1024,
    #     max_clip_length=MAX_CLIP_LENGTH)

    optimizer = optim.Adam(
        model.parameters(),
        lr=MAX_LR)

    scaler = GradScaler()

    def train(model, trainloader):
        model.train()

        batch = next(iter(trainloader))

        with torch.autocast(
                enabled=AMP,
                dtype=torch.bfloat16,
                device_type="cuda"):
            #padding_len = max(0, SEQ_LEN - test_inp.size(0))
            #n_test_inp = F.pad(test_inp, (0, padding_len))
            #batch = n_test_inp.unsqueeze(0)
            # print(batch.shape)
            loss = model(batch, return_loss=True)
            # loss = model(next(trainloader), return_loss=True)
            # loss.backward()
            return loss

    def test(model, test_loader, dataset):
        model.eval()

        batch = next(iter(test_loader))
        with torch.no_grad():
            loss = model(batch, return_loss = True)
            return loss
        
    pbar = tqdm(range(EPOCHS), mininterval=10., desc='training')
    batch_idx = 0
    for i in pbar:
        mem_gb = get_reserved_mem_gb()

        lr = get_lr(batch_idx) if DECAY_LR else MAX_LR

        for b in range(len(train_loader)):
            # Set LR
            lr = get_lr(batch_idx) if DECAY_LR else MAX_LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            loss = train(model, train_loader)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_idx += 1
            pbar.set_description(
                f"Reserved Memory (GB): {mem_gb}, loss: {loss.item()}, lr: {lr}, batch: {b}/{len(train_loader)}")

        # Validate every `n` steps (because it's time consuming)
        if i % VALIDATE_EVERY == 0:
            vloss = test(model, test_loader, dataset)
            print(f'validation loss: {vloss.item()}')
            pbar.set_description(
                f"Reserved Memory (GB): {mem_gb}, loss: {loss.item()}, vloss: {vloss.item()}, lr: {lr}")

        # Save best model every `n` steps. Set this to be high as the models are huge
        if vloss < best_val:
            best_val = vloss
            if SAVE:
                torch.save(
                    model.state_dict(),
                    f"./megabyte_{i}_{vloss}.pt")
                torch.save(
                    optimizer.state_dict(),
                    f"./megabyte_{i}_{vloss}_optim.pt")
                    
        if i % PRINT_INTERVAL == 0:
            print(f"Reserved Memory (GB): {mem_gb}, loss: {loss.item()}, lr: {lr}")
        pbar.set_description(f"Reserved Memory (GB): {mem_gb}, loss: {loss.item()}, lr: {lr}")

    item = next(iter(eval_loader))[0] # [0]
    # print("item.shape:", item.shape)
    print("Generative Prompt:", item[-4])
    item_phone_tokens = item[-2]
    item_audio_tokens = item[-1]
    # print("Item, Aud, Phone:", item_phone_tokens, item_audio_tokens)
    phone_prompt, _, _ = multi_encode(
        item_phone_tokens,
        item_audio_tokens,
        n_phone_tokens=len(dataset.phone_dict),
        n_audio_tokens=1024,
        max_clip_length=MAX_CLIP_LENGTH)
    prompt, sample = generate(model, phone_prompt)

    model.eval()

    try:
        out = generate_audio(
            sample,
            n_phone_tokens=len(dataset.phone_dict),
            n_audio_tokens=1024)
        
        # ETT, EAT ids: [1099, 1100]
        values = sample.cpu().numpy()
        ETT_S  = np.where(values == 1099)
        EAT_S  = np.where(values == 1100)

        print("ETT_S, EAT_S:", ETT_S, EAT_S)

        print(out)

    except Exception as e:
        print("Failure generating audio:", e)