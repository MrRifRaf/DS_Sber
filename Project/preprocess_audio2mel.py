''' This module preprocesses mel spectrograms for faster training
'''
import torch

from tacotron2.data_function import TextMelLoader


def parse_args(args):
    """ Add arguments
    """
    args.dataset_path = r'.\data'       # Path to dataset
    args.subset_size = None

    # Type of text cleaners for input text
    args.text_cleaners = ['english_cleaners']

    args.sampling_rate = 16000         # Sampling rate
    args.filter_length = 1024          # Filter length
    args.hop_length = 256              # Hop (stride) length
    args.win_length = 1024             # Window length
    args.mel_fmin = 0.0                # Minimum mel frequency
    args.mel_fmax = 8000.0             # Maximum mel frequency
    args.n_mel_channels = 80           # Number of bins in mel-spectrograms

    return args


def audio2mel(dataset_path, args):
    ''' Calculates and saves mel spectrogram to the disk
    '''

    data_loader = TextMelLoader(args.dataset_path, args.training_files, args)
    data_len = len(data_loader)

    for i in range(data_len):
        if i % 100 == 0:
            print(f"done {i:>5} / {data_len:>5}")

        filename = data_loader.libri._walker[i]
        mel = data_loader[i][1]
        torch.save(
            mel, rf'{dataset_path}\mel\{args.training_files}\{filename}.mel')


class Args:
    pass


def main():

    args = Args()
    args = parse_args(args)
    args.load_mel_from_disk = False
    args.training_files = 'test-clean'

    audio2mel(args.dataset_path, args)


if __name__ == '__main__':
    main()
