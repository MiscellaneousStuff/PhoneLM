from g2p_en import G2p

import torch
import random
import string
from functools import cache
from tqdm import tqdm

@cache
def _get_model():
    return G2p()

@cache
def _get_graphs(path):
    with open(path, "r") as f:
        graphs = f.read()
    return graphs

def encode_text(graphs: str) -> list[str]:
    g2p = _get_model()
    phones = g2p(graphs)
    ignored = {" ", *string.punctuation}
    return ["_" if p in ignored else p for p in phones]

def encode_text_direct(text):
    g2p = _get_model()
    phones = g2p(text)
    ignored = {" ", *string.punctuation}
    return ["_" if p in ignored else p for p in phones]

@torch.no_grad()
def write_phones(folder, suffix=".normalized.txt"):
    paths = list(folder.rglob(f"*{suffix}"))
    random.shuffle(paths)

    for path in tqdm(paths):
        phone_path = path.with_name(path.stem.split(".")[0] + ".phn.txt")
        if phone_path.exists():
            continue
        graphs = _get_graphs(path)
        phones = encode_text(graphs)
        with open(phone_path, "w") as f:
            f.write(" ".join(phones))