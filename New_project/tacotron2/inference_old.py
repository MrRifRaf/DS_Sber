import importlib
import os
from contextlib import contextmanager

import torch
from scipy.io import wavfile

from common.utils import remove_crackle
from router import models

configs = importlib.import_module('configs')
configs = importlib.reload(configs)

Config = configs.Config


def main():
    generate_wav_file(audio_path)


def generate_wav_file():
    audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'infer', 'result.wav')
    wg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnt',
                           'pretrained', 'wg_fp32_torch')
    mel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnt',
                            'train', 'Cori_Samuel', 'mels',
                            'beyondgoodandevil_02_nietzsche_0222.mel')
    sample, _ = save_sample(mel_path, wg_path)
    sample = remove_crackle(sample, Config.wdth, Config.snst)

    wavfile.write(audio_path, 22050, sample)
    return audio_path


@contextmanager
def evaluating(model):
    """
    Temporarily switch to evaluation mode.

    :param model:
    :return:
    """
    istrain = model.training
    try:
        model.eval()
        yield model
    finally:
        if istrain:
            model.train()


def restore_checkpoint(restore_path, model_name):
    """ :param restore_path:
        :param model_name:
        :return:
    """
    checkpoint = torch.load(restore_path, map_location='cpu')

    print(f'Restoring from `{restore_path}` checkpoint')

    model_config = checkpoint['config']
    model = models.get_model(model_name, model_config, to_cuda=False)

    # Unwrap distributed
    model_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace('module.1.', '')
        new_key = new_key.replace('module.', '')
        model_dict[new_key] = value

    model.load_state_dict(model_dict)

    return model


def save_sample(mel_path, model_path):
    """ :param mel_path:
        :param model_path:
        :return:
    """
    waveglow_path = model_path
    wg = restore_checkpoint(waveglow_path, 'WaveGlow')

    with evaluating(wg), torch.no_grad():
        mel = torch.load(mel_path, map_location='cpu')
        audio = wg.infer(mel.unsqueeze(0))
        audio_numpy = audio[0].data.cpu().numpy()

        return audio_numpy, mel


if __name__ == '__main__':
    main()
