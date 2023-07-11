"""
Modified version of torchaudio dataset class for LJSpeech.
Instead of returning the waveform, it uses encodec to return the audio file codes.
"""

import csv
import os
from pathlib import Path
from typing import Tuple, Union

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from g2p_en import G2p
from g2p_util import encode_text_direct, _get_model
from encodec_util import encode, decode

_RELEASE_CONFIGS = {
    "release1": {
        "folder_in_archive": "wavs",
        "url": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        "checksum": "be1a30453f28eb8dd26af4101ae40cbf2c50413b1bb21936cbcdc6fae3de8aa5",
    }
}

class LJSPEECH(Dataset):
    """*LJSpeech-1.1* :cite:`ljspeech17` dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found.
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"wavs"``).
    """

    def __init__(
        self,
        root: Union[str, Path],
        encodec_bandwidth: float = 6.0,
        folder_in_archive: str   = _RELEASE_CONFIGS["release1"]["folder_in_archive"],
        max_prompt_length: int   = 60
    ) -> None:

        self._parse_filesystem(root, folder_in_archive, max_prompt_length)
        self.encodec_bandwidth = encodec_bandwidth
        self.phone_dict = _get_model().phonemes + ["_"]

    def _parse_filesystem(
            self,
            root: str,
            folder_in_archive: str,
            max_prompt_length: int) -> None:
        
        root = Path(root)

        basename = os.path.basename(_RELEASE_CONFIGS["release1"]["url"])

        basename = Path(basename.split(".tar.bz2")[0])
        folder_in_archive = basename / folder_in_archive

        self._path = root / folder_in_archive
        self._metadata_path = root / basename / "metadata.csv"

        if not os.path.exists(self._path):
            raise RuntimeError(
                f"The path {self._path} doesn't exist. "
                "Please check the ``root`` path"
            )

        with open(self._metadata_path, "r", newline="", encoding="utf-8") as metadata:
            flist = csv.reader(metadata, delimiter="|", quoting=csv.QUOTE_NONE)
            self._flist = list(flist)
            self._flist = [item
                           for item in self._flist
                           if len(item[2]) <= max_prompt_length]
            # if max_prompt_length:
            #     self._flist = [item
            #                    for item in self._flist
            #                    if len(item[1]) <= max_prompt_length]
            #     for i in range(len(self._flist)):
            #         item              = self._flist[i][1]
            #         self._flist[i][1] = " ".join(item.split(" ")[0:max_prompt_length//6])

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Transcript
            str:
                Normalized Transcript
        """
        line = self._flist[n]
        fileid, transcript, normalized_transcript = line
        fileid_audio = self._path / (fileid + ".wav")

        # Load audio
        waveform, sample_rate = torchaudio.load(fileid_audio)

        # G2P and Encodec
        phones    = encode_text_direct(normalized_transcript)
        phone_ids = torch.tensor(
            [self.phone_dict.index(phone) for phone in phones]).long().cuda()
        codes     = encode(waveform, sample_rate, self.encodec_bandwidth)

        return (
            fileid_audio,
            waveform,
            sample_rate,
            transcript,
            normalized_transcript,
            phones,
            phone_ids,
            codes
        )

    def __len__(self) -> int:
        return len(self._flist)