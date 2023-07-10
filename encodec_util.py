
import random
import soundfile
from pathlib import Path

import torch
from torch import Tensor
import torchaudio

from functools import cache
from tqdm import tqdm
from einops import rearrange

from encodec import EncodecModel
from encodec.utils import convert_audio

SAMPLE_RATE = 24_000
BANDWIDTH = 6.0

@cache
def _load_model(bandwidth=6.0, device="cuda"):
    # Instantiate a pretrained EnCodec model
    assert SAMPLE_RATE == 24_000
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    model.to(device)
    return model

def unload_model():
    return _load_model.cache_clear()

@torch.inference_mode()
def decode(codes: Tensor, bandwidth=6.0, device="cuda"):
    """
    Args:
        codes: (b q t)
    """
    assert codes.dim() == 3
    model = _load_model(bandwidth, device)
    return model.decode([(codes, None)]), model.sample_rate

def decode_to_file(resps: Tensor, path: Path):
    assert resps.dim() == 2, f"Require shape (t q), but got {resps.shape}."
    resps = rearrange(resps, "t q -> 1 q t")
    wavs, sr = decode(codes=resps, bandwidth=BANDWIDTH)
    soundfile.write(str(path), wavs.cpu()[0, 0], sr)

def _replace_file_extension(path, suffix):
    return (path.parent / path.name.split(".")[0]).with_suffix(suffix)

@torch.inference_mode()
def encode(wav: Tensor, sr: int, bandwidth=6.0, device="cuda"):
    """
    Args:
        wav: (t)
        sr: int
    """
    model = _load_model(bandwidth, device)
    wav = wav.unsqueeze(0)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.to(device)
    encoded_frames = model.encode(wav)
    qnt = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # (b q t)
    return qnt

def encode_from_file(path, bandwidth=6.0, device="cuda"):
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] == 2:
        wav = wav[:1]
    return encode(wav, sr, bandwidth, device)

def quantize_audio(folder, suffix=".wav"):
    paths = [*folder.rglob(f"*{suffix}")]
    random.shuffle(paths)

    for path in tqdm(paths):
        out_path = _replace_file_extension(path, ".qnt.pt")
        if out_path.exists():
            continue
        qnt = encode_from_file(path, BANDWIDTH)
        print(qnt.shape)
        torch.save(qnt.cpu(), out_path)

def decode_files(folder, suffix=".qnt.pt"):
    paths = [*folder.rglob(f"*{suffix}")]
    random.shuffle(paths)

    for path in tqdm(paths):
        out_path = _replace_file_extension(path, ".qt.wav")
        if out_path.exists():
            continue
        fi = rearrange(torch.load(path).squeeze(0).cuda(), "q t -> t q")
        decode_to_file(fi, out_path)