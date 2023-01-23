''' This module converts audio to mel spectrograms
'''
import argparse
import importlib
import os
import shutil
import warnings

import torch

from common.utils import load_filepaths_and_text
from router import data_functions

warnings.filterwarnings('ignore')

# Parse args
parser = argparse.ArgumentParser(description='Pre-processing')
parser.add_argument('--exp',
                    type=str,
                    default=None,
                    required=True,
                    help='Name of an experiment for configs setting.')
args = parser.parse_args()

# Prepare config
shutil.copyfile(os.path.join('configs', 'experiments', args.exp + '.py'),
                os.path.join('configs', '__init__.py'))

# Reload Config
configs = importlib.import_module('configs')
configs = importlib.reload(configs)
Config = configs.Config


def audio2mel(audiopaths_and_text, melpaths_and_text):
    ''' Creates and saves mel-spectrograms from audio
    '''
    melpaths_and_text_list = load_filepaths_and_text(melpaths_and_text)
    audiopaths_and_text_list = load_filepaths_and_text(audiopaths_and_text)

    data_loader = data_functions.get_data_loader('Tacotron2',
                                                 audiopaths_and_text)

    size = len(melpaths_and_text_list)
    for i in range(size):
        if i % 100 == 0:
            print(f'done {i:>6d}/{size}')

        mel = data_loader.get_mel(audiopaths_and_text_list[i][0])
        torch.save(mel, melpaths_and_text_list[i][0])


def main():
    ''' Main function of a module
    '''
    for set_type in ['train', 'val']:
        print(f'Processing {set_type} set')
        Config.wav_files = os.path.join('mnt', 'train', set_type + '.txt')
        Config.mel_files = os.path.join('mnt', 'train', set_type + '_mel.txt')

        audio2mel(Config.wav_files, Config.mel_files)


if __name__ == '__main__':
    main()
