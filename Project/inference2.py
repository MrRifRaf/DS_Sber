# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import logging
import os
import sys
import time

import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from scipy.io.wavfile import write

import models
from tacotron2.text import text_to_sequence
from waveglow.denoiser import Denoiser


def parse_args(args):
    args.input = os.path.join('infer', 'input.txt')
    args.output = 'infer'
    args.suffix = ""

    args.tacotron2 = os.path.join('output', 'checkpoint_Tacotron2_last.pt')
    args.waveglow = os.path.join('output', 'waveglow_1076430_14000_amp')
    args.sigma_infer = 0.9
    args.denoising_strength = 0.01
    args.sampling_rate = 16000
    args.fp16 = False
    args.cpu = not args.fp16
    args.log_file = 'infer.log'
    args.include_warmup = False
    args.stft_hop_length = 256
    args.n_speakers = 2
    args.speakers_embedding_dim = 2
    return args


def checkpoint_from_distributed(state_dict):
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


def load_and_setup_model(model_name,
                         parser,
                         checkpoint,
                         fp16_run,
                         cpu_run,
                         forward_is_infer=False):
    model_parser = models.model_parser(model_name, parser, add_help=False)

    model_config = models.get_model_config(model_name, model_parser)
    model = models.get_model(model_name,
                             model_config,
                             cpu_run=cpu_run,
                             forward_is_infer=forward_is_infer)

    if checkpoint is not None:
        if cpu_run:
            state_dict = torch.load(
                checkpoint, map_location=torch.device('cpu'))['state_dict']
        else:
            state_dict = torch.load(checkpoint)['state_dict']
        if checkpoint_from_distributed(state_dict):
            state_dict = unwrap_distributed(state_dict)

        model.load_state_dict(state_dict)

    if model_name == "WaveGlow":
        model = model.remove_weightnorm(model)

    model.eval()

    if fp16_run:
        model.half()

    return model


# taken from tacotron2/data_function.py:TextMelCollate.__call__
def pad_sequences(batch):
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor(
        [len(x) for x in batch]),
                                                      dim=0,
                                                      descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]
        text_padded[i, :text.size(0)] = text

    return text_padded, input_lengths


def prepare_input_sequence(texts, speaker_sexes, cpu_run=False):

    d = []
    for text in texts:
        d.append(
            torch.IntTensor(
                text_to_sequence('~' + text + '@', ['english_cleaners'])[:]))

    speaker_sexes = torch.IntTensor(speaker_sexes)
    text_padded, input_lengths = pad_sequences(d)

    if not cpu_run:
        text_padded = text_padded.cuda().long()
        input_lengths = input_lengths.cuda().long()
        speaker_sexes = speaker_sexes.cuda().long()
    else:
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()
        speaker_sexes = speaker_sexes.long()

    return text_padded, input_lengths, speaker_sexes


class MeasureTime():

    def __init__(self, measurements, key, cpu_run=False):
        self.measurements = measurements
        self.key = key
        self.cpu_run = cpu_run

    def __enter__(self):
        if not self.cpu_run:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.cpu_run:
            torch.cuda.synchronize()
        self.measurements[self.key] = time.perf_counter() - self.t0


class Args:

    def __init__(self):
        pass


def main():
    args = Args()
    args = parse_args(args)

    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    file_handler = logging.FileHandler(args.output + '/' + args.log_file,
                                       mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler_format = '%(asctime)s | %(levelname)s | %(lineno)d: %(message)s'
    file_handler.setFormatter(logging.Formatter(file_handler_format))
    logger.addHandler(file_handler)

    for k, v in vars(args).items():
        logger.info(dict(step="PARAMETER", data={k: v}))
    logger.info(dict(step="PARAMETER", data={'model_name': 'Tacotron2_PyT'}))

    tacotron2 = load_and_setup_model('Tacotron2',
                                     args,
                                     args.tacotron2,
                                     args.fp16,
                                     args.cpu,
                                     forward_is_infer=True)
    waveglow = load_and_setup_model('WaveGlow',
                                    args,
                                    args.waveglow,
                                    args.fp16,
                                    args.cpu,
                                    forward_is_infer=True)
    denoiser = Denoiser(waveglow)
    if not args.cpu:
        denoiser.cuda()

    jitted_tacotron2 = torch.jit.script(tacotron2)

    texts, speaker_sexes = [], []
    try:
        f = open(args.input, 'r', encoding='utf-8')
        for line in f:
            text, sex = line.split('|')
            text, sex = text.strip(), sex.strip()
            texts.append(text)
            speaker_sexes.append(1 if sex == 'M' else 0)
        f.close()
    except:
        print("Could not read file")
        sys.exit(1)

    if args.include_warmup:
        sequence = torch.randint(low=0, high=148, size=(1, 50)).long()
        input_lengths = torch.IntTensor([sequence.size(1)]).long()
        if not args.cpu:
            sequence = sequence.cuda()
            input_lengths = input_lengths.cuda()
        for i in range(3):
            with torch.no_grad():
                mel, mel_lengths, _ = jitted_tacotron2(sequence, input_lengths)
                _ = waveglow(mel)

    measurements = {}

    sequences_padded, input_lengths, speaker_sexes = prepare_input_sequence(
        texts, speaker_sexes, args.cpu)

    with torch.no_grad(), MeasureTime(measurements, "tacotron2_time",
                                      args.cpu):
        mel, mel_lengths, alignments = jitted_tacotron2(
            sequences_padded, input_lengths, speaker_sexes)

    with torch.no_grad(), MeasureTime(measurements, "waveglow_time", args.cpu):
        audios = waveglow(mel, sigma=args.sigma_infer)
        audios = audios.float()
    with torch.no_grad(), MeasureTime(measurements, "denoiser_time", args.cpu):
        audios = denoiser(audios, strength=args.denoising_strength).squeeze(1)

    print("Stopping after", mel.size(2), "decoder steps")
    tacotron2_infer_perf = mel.size(0) * mel.size(
        2) / measurements['tacotron2_time']
    waveglow_infer_perf = audios.size(0) * audios.size(
        1) / measurements['waveglow_time']

    logger.info(
        dict(step=0, data={"tacotron2_items_per_sec": tacotron2_infer_perf}))
    logger.info(
        dict(step=0,
             data={"tacotron2_latency": measurements['tacotron2_time']}))
    logger.info(
        dict(step=0, data={"waveglow_items_per_sec": waveglow_infer_perf}))
    logger.info(
        dict(step=0, data={"waveglow_latency": measurements['waveglow_time']}))
    logger.info(
        dict(step=0, data={"denoiser_latency": measurements['denoiser_time']}))
    logger.info(
        dict(step=0,
             data={
                 "latency": (measurements['tacotron2_time'] +
                             measurements['waveglow_time'] +
                             measurements['denoiser_time'])
             }))
    for i, audio in enumerate(audios):

        align_path = os.path.join(args.output,
                                  "alignment_" + str(i) + args.suffix + ".png")
        plt.imshow(alignments[i].float().data.cpu().numpy().T,
                   aspect="auto",
                   origin="lower")
        plt.savefig(align_path)

        audio = audio[:mel_lengths[i] * args.stft_hop_length]
        audio = audio / torch.max(torch.abs(audio))
        audio_path = os.path.join(args.output,
                                  "audio_" + str(i) + args.suffix + ".wav")
        write(audio_path, args.sampling_rate, audio.cpu().numpy())

    for i, m in enumerate(mel):
        mel_path = os.path.join(args.output,
                                "mel_" + str(i) + args.suffix + ".png")
        librosa.display.specshow(m.cpu().numpy())
        plt.savefig(mel_path)

    file_handler.flush()


if __name__ == '__main__':
    main()
