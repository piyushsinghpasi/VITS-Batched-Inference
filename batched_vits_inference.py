import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time
import pandas as pd
# '2.1.2+cu121'
import sys
sys.path.insert(0, '..')

import commons
import utils

from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from torch.nn.utils.rnn import pad_sequence

from scipy.io.wavfile import write as write_audio


import argparse
from typing import List


parser = argparse.ArgumentParser(description="generate TTS samples from VITS TTS")

parser.add_argument("--vits_config", type=str,
    default="../configs/vctk_base.json",
    help="vits config path",
)
parser.add_argument("--vits_checkpoint", type=str,
    default="./pretrained_ljs.pth",
    help="vits checkpoint path",
)
parser.add_argument("--audio_saving_dir", "-v", type=str,
    default="./VITS_TTS_samples/",
    help="path where tts samples from VITS model will be saved"
)
parser.add_argument("--data_file", type=str,
    default="./test_data.csv",
    help="path to data csv or tsv containing `text` and `audio_filename` columns"
        "`text` is the text for which audio will be generated"
        "`audio_filename` is path where generated audio will be saved"
)
parser.add_argument("--seed", type=int,
    default=1, help="seed for reproducibility"
)
parser.add_argument("--start_idx", type=int,
    default=0,
    help="start from this idx in dataframe used for multiprocessing (inclusive)"
)
parser.add_argument("--end_idx", type=int,
    default=None,
    help="end on this idx in dataframe used for multiprocessing (NOT inclusive, so range is [start_idx, end_idx) )"
        "If not passed will default to full length"
)
parser.add_argument("--batch_size", type=int,
    help="batch size"
)
parser.add_argument("--noise_scale", type=float,
    default=.667,
    help="noise scale used for inference"
)
parser.add_argument("--noise_scale_w", type=float,
    default=.8,
    help="noise scale w used for inference"
)
parser.add_argument("--length_scale", type=float,
    default=1,
    help="duration used for inference"
)

args = parser.parse_args()
print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
print("-"*100)

torch.manual_seed(args.seed)

os.makedirs(args.audio_saving_dir, exist_ok=True)


def load_file(file):
    if file.endswith(".tsv"):
        df = pd.read_csv(file, sep="\t")
    else:
        df = pd.read_csv(file)
    return df

df = load_file(args.data_file)

columns = df.columns
assert "text" in columns, "csv/tsv file must have column called `text` on which TTS will be performed"
assert "audio_filename" in columns, \
    "csv/tsv file must have column called `audio_filename` where audio will be saved (audio_saving_dir/audio_filename)"

if args.end_idx is None:
    args.end_idx = len(df)

print(f"Data used from index {args.start_idx} to index {args.end_idx}")

df = df.iloc[args.start_idx:args.end_idx].reset_index(drop=True)

print("Data on which generation will happen")
print(df)


texts, audio_save_files = df["text"].tolist(), df["audio_filename"].tolist()

@torch.no_grad()
def batched_inference(x_tst:torch.Tensor, x_tst_lengths:torch.Tensor):
    """performs batched inference on tokenized text and saves it in file.

    Args:
        x_tst (torch.Tensor): tokenized and padded text 
        x_tst_lengths (torch.Tensor): length of text

    Returns:
        (torch.Tensor, torch.Tensor): audio & its length
    """

    x_tst = x_tst.cuda()
    x_tst_lengths = x_tst_lengths.cuda()

    audios, _, mask, *_ = net_g_vits.infer(
        x_tst, 
        x_tst_lengths, 
        noise_scale=args.noise_scale, 
        noise_scale_w=args.noise_scale_w, 
        length_scale=args.length_scale,
    )
    audio_lens = mask.sum([1,2]).long() * hps_vits.data.hop_length


    return audios, audio_lens


def write_batched_audios(audios: torch.Tensor, audio_lens: torch.Tensor, batch_save_file_name: List[str]):
    """write audios to wav files

    Args:
        audios (torch.Tensor): generated audio
        audio_lens (torch.Tensor): length of generated audio
        batch_save_file_name (List[str]): list of saving path 
            (1 for each audio, len(batch_save_file_name)==batch_size)
    """
    for audio, audio_len, audio_save_path in zip(audios, audio_lens, batch_save_file_name):
        audio = audio[0][:audio_len]
        audio = audio.cpu().numpy()

        write_audio(audio_save_path, rate=hps_vits.data.sampling_rate, data=audio)   
        
        print(f"Saved to: {audio_save_path}")

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


############ VITS ############################################################################
hps_vits = utils.get_hparams_from_file(args.vits_config)

    
net_g_vits = SynthesizerTrn(
    len(symbols),
    hps_vits.data.filter_length // 2 + 1,
    hps_vits.train.segment_size // hps_vits.data.hop_length,
    **hps_vits.model).cuda()
_ = net_g_vits.eval()

_ = utils.load_checkpoint(args.vits_checkpoint, net_g_vits, None)

start_time = time.time()

saved_tts_abs_path = []

curr_text_batch = []
curr_text_len_batch = []
batch_save_file_name = []



for idx, (text, audio_save_file) in enumerate(zip(texts, audio_save_files)):

    print(f"Processing text: {text}")

    audio_save_path = os.path.join(args.audio_saving_dir, audio_save_file.replace(".wav", "")+".wav")

    if os.path.exists(audio_save_path) and (idx != (len(texts)-1)):
        # path exists
        # we are not on last file
        print(f"Audio already exists: {audio_save_path}")
        continue
    elif os.path.exists(audio_save_path) and idx == len(texts)-1:
        # if on last file
        # but it already exists
        # do nothing and run for previous texts left
        pass
    else:
        # audio doesnt exist, 
        # generate for it
        stn_tst_vits = get_text(text, hps_vits)
        curr_text_batch.append(stn_tst_vits)
        curr_text_len_batch.append(stn_tst_vits.size(0))


        batch_save_file_name.append(
            audio_save_path
        )


    if ((idx+1) % args.batch_size == 0)  or (len(curr_text_batch)>0 and (idx == len(texts)-1)):
        # either batch-size achieved
        # or last batch (can be of diff length)
        x_tst = pad_sequence(curr_text_batch, batch_first=True)
        x_tst_lengths = torch.LongTensor(curr_text_len_batch)
    else:
        # keep accumulating until we hit batch size or last batch
        continue

    # generate audio
    audios, audio_lens = batched_inference(x_tst, x_tst_lengths)

    # write audio
    write_batched_audios(audios, audio_lens, batch_save_file_name)
        
    # clear cache
    torch.cuda.empty_cache()

    # reset curr batch
    curr_text_batch = []
    curr_text_len_batch = []
    batch_save_file_name = []




print(f"Total time: {(time.time()-start_time)/60 :.2f} mins")
print(f"Vits Samples saved at: {args.audio_saving_dir}")
