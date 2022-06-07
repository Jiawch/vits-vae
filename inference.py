from train_fs import main
import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import argparse

from scipy.io.wavfile import write
from tqdm import tqdm

import sys, os
sys.path.append('hifigan/')
from env import AttrDict
from vocoder_models import Generator as HiFiGAN

print('Initializing HiFi-GAN...')
with open('hifigan_pretrained/config.json') as f:
    h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load('hifigan_pretrained/generator_v1', map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def main(args):
    hps = utils.get_hparams_from_file(args.c)
    if hps.data.n_speakers > 0:
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model).cuda()
        _ = net_g.eval()
    else:
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model).cuda()
        _ = net_g.eval()

    utils.load_checkpoint(args.ckpt, net_g, None)
    
    if hps.data.n_speakers > 0:
        test_file = "filelists/vctk_audio_sid_text_test_filelist.txt"
    else:
        test_file = "filelists/ljs_audio_text_test_filelist.txt"
    output_dir = args.o
    os.makedirs(output_dir, exist_ok=True)

    if hps.data.n_speakers == 0:
        for line in open(test_file, "r").readlines():
            item_id, txt = line.strip().split("|")
            # print(item_id, txt)
            stn_tst = get_text(txt, hps)
            with torch.no_grad():
                x_tst = stn_tst.cuda().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
                mel = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0]     # [b, h, t]
                audio = vocoder.forward(mel)[0, 0].cpu().numpy()    # [t * hop]
                output_file = os.path.join(output_dir, os.path.basename(item_id) +'.wav')
                write(output_file, 22050, audio)
    else:
        vctk = [x.strip() for x in open("filelists/vctk.txt", "r").readlines()]
        for line in tqdm(open(test_file, "r").readlines()):
            item_id, sid, txt = line.strip().split("|")
            item_id = os.path.splitext(item_id.split("/")[-1])[0]
            if item_id not in vctk:
                continue
            stn_tst = get_text(txt, hps)
            with torch.no_grad():
                x_tst = stn_tst.cuda().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
                sid = torch.LongTensor([int(sid)]).cuda()
                audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
                output_file = os.path.join(output_dir, item_id +'.wav')
                write(output_file, 22050, audio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", default="config/lj_base.json")
    parser.add_argument("-o", default="output")
    parser.add_argument("-m", default="vits_exp1")
    args = parser.parse_args()
    log_dir = "logs"
    args.model_dir = os.path.join(log_dir, args.m)
    args.ckpt = utils.latest_checkpoint_path(os.path.join(args.model_dir, "ckpt"), "G_*.pth")
    global_step = os.path.splitext(os.path.basename(args.ckpt).split("_")[1])[0]
    args.o = os.path.join(args.model_dir, f"{args.o}_{global_step}")
    main(args)