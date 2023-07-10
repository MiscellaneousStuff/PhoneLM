
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

import random
import numpy as np

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

MAX_LR       = 1e-2
# MAX_LR       = 1e-2
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 0.1

BANDWIDTH_IDX = 0 # original VALL-E
CODEBOOKS     = [2, 4, 8, 16, 32]
BANDWIDTHS    = [1.5, 3.0, 6.0, 12.0, 24.0]
BANDWIDTH     = BANDWIDTHS[BANDWIDTH_IDX]
CODEBOOK      = CODEBOOKS[BANDWIDTH_IDX]

def get_reserved_mem_gb():
    device = torch.cuda.current_device()
    reserved = torch.cuda.memory_reserved(device)
    reserved_gb = reserved / 1024 / 1024 / 1024
    return reserved_gb

class PhoneLM(nn.Module):
    def __init__(self, n_phone_tokens, n_audio_tokens):
        super(PhoneLM, self).__init__()
        self.megabyte   = megabyte.MEGABYTE(
            heads       = 8, # 1,
            dim_head    = 32, # 16,
            num_tokens  = n_phone_tokens + n_audio_tokens + 4,
            dim         = (768, 256, 128), # (32, 32, 32), # (768, 256, 128)# Dg, Dl1, Dl2
            depth       = (6, 4, 2), # (6, 4, 2)
            max_seq_len = (32, 4, 4), # (128, 4, 4), # (32, 4, 4), # 512
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
    STT, ETT, SAT, EAT = [n_phone_tokens + n_audio_tokens + i
                          for i in range(4)]
    STT = torch.tensor([STT]).long().cuda()
    ETT = torch.tensor([ETT]).long().cuda()
    SAT = torch.tensor([SAT]).long().cuda()
    EAT = torch.tensor([EAT]).long().cuda()

    if max_clip_length:
        print("pre audio_tokens.shape", audio_tokens.shape)
        audio_tokens = audio_tokens[:, :, :int(max_clip_length * 75)]
    print("post audio_tokens.shape", audio_tokens.shape)
    audio_tokens = rearrange(audio_tokens.squeeze(0), "q s -> (q s)")
    print("post einops audio_tokens.shape", audio_tokens.shape)
    
    # offset phone tokens past audio tokens
    phone_tokens += n_audio_tokens
    
    print("phone_tokens.shape:", phone_tokens.shape)
    print("audio_tokens.shape:", audio_tokens.shape)
    
    device = torch.cuda.current_device()
    phone_tokens = torch.cat((STT, phone_tokens, ETT), dim=0).to(device)
    audio_tokens = torch.cat((SAT, audio_tokens, EAT,), dim=0).to(device)
    combined_tokens = torch.cat((phone_tokens, audio_tokens), dim=0).to(device)
    return phone_tokens, audio_tokens, combined_tokens

def generate_audio(sample,
                   n_phone_tokens,
                   n_audio_tokens,
                   audio_path="./out.wav"):
    STT, ETT, SAT, EAT = [n_phone_tokens + n_audio_tokens + i
                          for i in range(4)]
    ST_S = [STT, ETT, SAT, EAT]
    print("STT, ETT, SAT, EAT ids:", ST_S)
    seq = sample.cpu().tolist()[0]
    # print("seq:", seq)
    # all special tokens in list
    if all(st_t in seq for st_t in ST_S) and len(seq) >= len(ST_S) + 2:
        # text_tokens  = seq[seq.index(STT + 1):seq.index(ETT - 1)]
        audio_tokens = seq[seq.index(SAT)+1:seq.index(EAT)]
        print(seq.index(SAT), seq.index(EAT), len(audio_tokens))
        audio_tokens = torch.tensor(audio_tokens).cuda()
        audio_tokens = rearrange(
            audio_tokens,
            '(q t) -> t q',
            q=CODEBOOK,
            t=audio_tokens.size(0) // CODEBOOK)
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

if __name__ == "__main__":
    dataset = LJSPEECH("./data/LJSpeech",
                       encodec_bandwidth=BANDWIDTH)

    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler, collate_fn=lambda x: x)
    test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler, collate_fn=lambda x: x)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhoneLM(
        n_phone_tokens=len(dataset.phone_dict),
        n_audio_tokens=1024).to(device)

    print("Model params:", model.megabyte.get_num_params())

    item = next(iter(train_loader))[0]
    item_phone_tokens = item[-2]
    item_audio_tokens = item[-1]
    item_phone_tokens.shape, item_audio_tokens.shape
    print(item[0], item[3])

    print("item_audio_tokens:", item_audio_tokens)

    phone_prompt, audio_target, test_inp = multi_encode(
        item_phone_tokens,
        item_audio_tokens,
        n_phone_tokens=len(dataset.phone_dict),
        n_audio_tokens=1024,
        max_clip_length=1.5)
    

    optimizer = optim.Adam(
        model.parameters(),
        lr=MAX_LR)

    EPOCHS = 500
    PRINT_INTERVAL = 100

    seq_len = 512

    def train(model, trainloader):
        model.train()
        
        padding_len = max(0, seq_len - test_inp.size(0))
        n_test_inp = F.pad(test_inp, (0, padding_len))
        batch = n_test_inp.unsqueeze(0)
        # print(batch.shape)
        loss = model(batch, return_loss=True)
        # loss = model(next(trainloader), return_loss=True)
        loss.backward()
        return loss

    pbar = tqdm(range(EPOCHS), mininterval=10., desc='training')
    for i in pbar:
        loss = train(model, train_loader)
        optimizer.step()
        optimizer.zero_grad()
        mem_gb = get_reserved_mem_gb()
        if i % PRINT_INTERVAL == 0:
            print(f"Reserved Memory (GB): {mem_gb}, loss: {loss.item()}")
        pbar.set_description(f"Reserved Memory (GB): {mem_gb}, loss: {loss.item()}")

    prompt, sample = generate(model, phone_prompt)

    out = generate_audio(
        sample,
        n_phone_tokens=len(dataset.phone_dict),
        n_audio_tokens=1024)
    
    # STT, ETT, SAT, EAT ids: [1099, 1100, 1101, 1102]
    values = sample.cpu().numpy()
    STT_S  = np.where(values == 1099)
    ETT_S  = np.where(values == 1100)
    SAT_S  = np.where(values == 1101)
    EAT_S  = np.where(values == 1102)

    print("STT_S, ETT_S, SAT_S, EAT_S:", STT_S, ETT_S, SAT_S, EAT_S)

    print(out)