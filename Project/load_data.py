import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from torchaudio import load
from torchaudio.datasets import LIBRISPEECH

from tacotron2.arg_parser import tacotron2_parser
from tacotron2.data_function import TextMelLoader
from tacotron2_common.utils import load_wav_to_torch
from waveglow.arg_parser import waveglow_parser
from waveglow.data_function import MelAudioLoader

data = LIBRISPEECH(r'C:\Users\gabdullin-rr\Tuition\Python\DS_Sber\Project\data',
                   url='train-clean-100', folder_in_archive='LibriSpeech', download=True)

train_data = LIBRISPEECH(r'C:\Users\gabdullin-rr\Tuition\Python\DS_Sber\Project\data',
                         url='test-clean', folder_in_archive='LibriSpeech', download=True)
train_data._walker[0]


torch.std(train_data[200][0]).item()
data[20][0].permute(1, 0).size()
bla = load(
    r'C:\Users\gabdullin-rr\Tuition\Python\DS_Sber\Project\data\LibriSpeech\train-clean-100\89\218\89-218-0057.flac')
bla[0].shape
bla[0].dtype
len(data[20][2])

len(data)


def parse_args(arguments):
    """ Train hyperparameters.
    """

    arguments.output = './output'           # Directory to save checkpoints
    arguments.dataset_path = './data'       # Path to dataset
    arguments.model_name = 'tacotron2'      # Model to train
    arguments.log_file = 'nvlog.json'       # Filename for logging
    arguments.epochs = 20                   # Number of total epochs to run

    # Epochs after which decrease learning rate
    arguments.anneal_steps = arguments.epochs // 2

    # Factor for annealing learning rate')
    arguments.anneal_factor = 0.1

    # parser.add_argument('--config-file', action=ParseFromConfigFile,
    #                     type=str, help='Path to configuration file')

    arguments.seed = None                   # Seed for random number generators

    # training
    arguments.epochs_per_checkpoint = 50    # Number of epochs per checkpoint
    arguments.checkpoint_path = ''          # Checkpoint path to resume training

    # Resumes training from the last checkpoint;
    # uses the directory provided with \'--output\' option
    # to search for the checkpoint \"checkpoint_<model_name>_last.pt\"')
    arguments.resume_from_last = True

    arguments.dynamic_loss_scaling = True   # Enable dynamic loss scaling
    arguments.amp = True                    # Enable AMP
    arguments.cudnn_enabled = True          # Enable cudnn
    arguments.cudnn_benchmark = True        # Run cudnn benchmark

    # disable uniform initialization of batchnorm layer weight
    arguments.disable_uniform_initialize_bn_weight = True

    # Optimization parameters
    arguments.sampling_rate = 16000         # Sampling rate
    arguments.filter_length = 1024          # Filter length
    arguments.hop_length = 256              # Hop (stride) length
    arguments.win_length = 1024             # Window length
    arguments.mel_fmin = 0.0                # Minimum mel frequency
    arguments.mel_fmax = 8000.0             # Maximum mel frequency
    arguments.load_mel_from_disk = False    # Load mel, or create it on the fly
    arguments.mask_padding = False          # Use mask padding
    arguments.n_mel_channels = 80           # Number of bins in mel-spectrograms

    # Type of text cleaners for input text
    arguments.text_cleaners = ['english_cleaners']

    arguments.subset_size = 16 * 90         # Size of dataset to use for train/test
    arguments.subset_size = None         # Size of dataset to use for train/test

    arguments.sampling_rate = 16000         # Sampling rate
    arguments.filter_length = 1024          # Filter length
    arguments.hop_length = 256              # Hop (stride) length
    arguments.win_length = 1024             # Window length
    arguments.mel_fmin = 0.0                # Minimum mel frequency
    arguments.mel_fmax = 8000.0             # Maximum mel frequency

    # Segment length (audio samples) processed per iteration
    args.segment_length = 8000

    return arguments


def draw_mel(mel):
    mel = mel.numpy()
    librosa.display.specshow(mel, sr=16000, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()


class Args:
    def __init__(self):
        pass


args = Args()
args = tacotron2_parser(args, None)
args = parse_args(args)


loader = MelAudioLoader(
    r'.\data', 'train-clean-100', args)
loader[4][1].shape
sample = loader[8]
sample[1].shape
plt.plot(sample[1].numpy())
plt.show()
draw_mel(sample[0])
tac_loader = TextMelLoader(
    r'.\data', 'train-clean-100', args)
tac_loader[8][0]
tac_loader[8][1].shape
draw_mel(tac_loader[8][1])
draw_mel(loader[4][0])
loader = TextMelLoader(
    r'C:\Users\gabdullin-rr\Tuition\Python\DS_Sber\Project\data', 'test-clean', args)
loader[0]
for i in range(8):
    print(loader[69*8 + i][0].shape)
loader[69*8 + 0]
for i in train_data:
    if len(i[])
train_data[69*8 + 0][2]

len(loader)
loader[7][0].size()
mel = loader[30][0].numpy()

print(args.text_cleaners)
