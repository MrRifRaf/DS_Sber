import json
import os

import numpy as np

from tacotron2.text import symbols

global symbols


class Config:
    # ** Audio params **
    sampling_rate = 22050  # Sampling rate
    filter_length = 1024  # Filter length
    hop_length = 256  # Hop (stride) length
    win_length = 1024  # Window length
    mel_fmin = 0.0  # Minimum mel frequency
    mel_fmax = 8000.0  # Maximum mel frequency
    n_mel_channels = 80  # Number of bins in mel-spectrograms
    max_wav_value = 32768.0  # Maximum audiowave value

    # Audio postprocessing params
    snst = 0.00005  # filter sensitivity
    wdth = 1000  # width of filter

    # ** Tacotron Params **
    # Symbols
    # Number of symbols in dictionary
    n_symbols = len(symbols)
    symbols_embedding_dim = 512  # Text input embedding dimension

    # Speakers
    n_speakers = 10  # Number of speakers
    speakers_embedding_dim = 16  # Speaker embedding dimension
    try:
        # Dict with speaker coefficients
        with open(os.path.join('mnt', 'train', 'speaker_coefficients.json'),
                  encoding='utf-8') as fin:
            speaker_coefficients = json.load(fin)
    except IOError:
        print('Speaker coefficients dict is not available')
        speaker_coefficients = None

    # Emotions
    use_emotions = False  # Use emotions
    n_emotions = 15  # N emotions
    emotions_embedding_dim = 8  # Emotion embedding dimension
    try:
        # Dict with emotion coefficients
        with open(os.path.join('mnt', 'train', 'emotion_coefficients.json'),
                  encoding='utf-8') as fin:
            emotion_coefficients = json.load(fin)
    except IOError:
        print('Emotion coefficients dict is not available')
        emotion_coefficients = None

    # Encoder
    encoder_kernel_size = 5  # Encoder kernel size
    encoder_n_convolutions = 3  # Number of encoder convolutions
    encoder_embedding_dim = 512  # Encoder embedding dimension

    # Attention
    attention_rnn_dim = 1024  # Number of units in attention LSTM
    # Dimension of attention hidden representation
    attention_dim = 128

    # Attention location
    # Number of filters for location-sensitive attention
    attention_location_n_filters = 32
    # Kernel size for location-sensitive attention
    attention_location_kernel_size = 31

    # Decoder
    n_frames_per_step = 2  # Number of frames processed per step
    max_frames = 2000  # Maximum number of frames for decoder
    decoder_rnn_dim = 1024  # Number of units in decoder LSTM
    prenet_dim = 256  # Number of ReLU units in prenet layers
    # Maximum number of output mel spectrograms
    max_decoder_steps = int(max_frames / n_frames_per_step)
    gate_threshold = 0.5  # Probability threshold for stop token
    # Dropout probability for attention LSTM
    p_attention_dropout = 0.1
    # Dropout probability for decoder LSTM
    p_decoder_dropout = 0.1
    # Stop decoding once all samples are finished
    decoder_no_early_stopping = False

    # Postnet
    postnet_embedding_dim = 512  # Postnet embedding dimension
    postnet_kernel_size = 5  # Postnet kernel size
    postnet_n_convolutions = 5  # Number of postnet convolutions

    # Optimization
    mask_padding = False  # Use mask padding
    use_loss_coefficients = False  # Use balancing coefficients
    # Loss scale for coefficients
    if emotion_coefficients is not None and speaker_coefficients is not None:
        loss_scale = 1.5 / (np.mean(list(speaker_coefficients.values())) *
                            np.mean(list(emotion_coefficients.values())))
    else:
        loss_scale = None

    # ** Waveglow params **
    n_flows = 12  # Number of steps of flow
    # Number of samples in a group processed by the steps of flow
    n_group = 8
    # Determines how often (i.e., after how many coupling layers)
    # a number of channels (defined by --early-size parameter)
    # are output to the loss function
    n_early_every = 4
    # Number of channels output to the loss function
    n_early_size = 2
    # Standard deviation used for sampling from Gaussian
    wg_sigma = 1.0
    # Segment length (audio samples) processed per iteration
    segment_length = 4000
    wn_config = dict(
        n_layers=8,  # Number of layers in WN
        # Kernel size for dialted convolution in the affine coupling layer (WN)
        kernel_size=3,
        n_channels=512 // 2  # Number of channels in WN
    )

    # ** Script args **
    model_name = "Tacotron2"
    # Directory to save checkpoints
    output_directory = os.path.join('mnt', 'logs')
    log_file = "logs.txt"  # Filename for logging

    # Epochs after which decrease learning rate
    anneal_steps = [500, 1000, 1500]
    anneal_factor = 0.1  # Factor for annealing learning rate

    # Path to pre-trained Tacotron2 checkpoint for sample generation
    tacotron2_checkpoint = os.path.join('mnt', 'pretrained', 't2_fp32_torch')
    # Path to pre-trained WaveGlow checkpoint for sample generation
    waveglow_checkpoint = os.path.join('mnt', 'pretrained', 'wg_fp32_torch')
    # Checkpoint path to restore from
    restore_from = ''

    # Training params
    epochs_per_checkpoint = 2  # 50  # Number of epochs per checkpoint
    epochs = 2 * epochs_per_checkpoint + 1  # 1501  # Number of total epochs to run
    # Seed for PyTorch random number generators
    seed = 1234
    dynamic_loss_scaling = True  # Enable dynamic loss scaling
    # Enable AMP (FP16) # TODO: Make it work
    amp_run = False
    cudnn_enabled = True  # Enable cudnn
    cudnn_benchmark = False  # Run cudnn benchmark

    # Optimization params
    use_saved_learning_rate = False
    learning_rate = 1e-3  # Learning rate
    weight_decay = 1e-6  # Weight decay
    grad_clip_thresh = 1  # Clip threshold for gradients
    batch_size = 32  # 64  # Batch size per GPU

    # Dataset
    # Loads mel spectrograms from disk instead of computing them on the fly
    load_mel_from_disk = True
    # Type of text cleaners for input text
    text_cleaners = ['english_cleaners']
    # Path to training filelist
    training_files = os.path.join('mnt', 'train', 'train_mel.txt')
    # Path to validation filelist
    validation_files = os.path.join('mnt', 'train', 'val_mel.txt')

    # Url used to set up distributed training
    dist_url = 'tcp://localhost:23456'
    group_name = "group_name"  # Distributed group name
    dist_backend = "nccl"  # Distributed run backend

    # Sample phrases
    phrases = {
        'speaker_ids': [1, 5],
        'texts': [
            'Hello, how are you doing today?',
            'I would like to eat a Hamburger.', 'Hi.',
            'I would like to eat a Hamburger. Would you like to join me?',
            'Do you have any hobbies?'
        ]
    }


class PreprocessingConfig:
    cpus = 4  # Amount of cpus for parallelization
    sr = 22050  # sampling ratio for audio processing
    top_db = 40  # level to trim audio
    # speaker to measure text_limit, dur_limit
    limit_by = 'Cori_Samuel'
    minimum_viable_dur = 0.05  # min duration of audio
    # max text length (used by default)
    text_limit = None
    # max audio duration (used by default)
    dur_limit = None
    n = 15000  # max size of training dataset per speaker
    # load data.csv - should be in output_directory
    start_from_preprocessed = True

    output_directory = os.path.join('mnt', 'train')
    raw_data = os.path.join('mnt', 'raw-data')
    data = [
        {
            'path': os.path.join(raw_data, 'Cori_Samuel'),
            'speaker_id': 0,  # 92,
            'process_audio': True,
            'emotion_present': False
        },
        {
            'path': os.path.join(raw_data, 'Phil_Benson'),
            'speaker_id': 1,  # 6097,
            'process_audio': True,
            'emotion_present': False
        },
        {
            'path': os.path.join(raw_data, 'John_Van_Stan'),
            'speaker_id': 2,  # 9017,
            'process_audio': True,
            'emotion_present': False
        },
        {
            'path': os.path.join(raw_data, 'Mike_Pelton'),
            'speaker_id': 3,  # 6670,
            'process_audio': True,
            'emotion_present': False
        },
        {
            'path': os.path.join(raw_data, 'Tony_Oliva'),
            'speaker_id': 4,  # 6671,
            'process_audio': True,
            'emotion_present': False
        },
        {
            'path': os.path.join(raw_data, 'Maria_Kasper'),
            'speaker_id': 5,  # 8051,
            'process_audio': True,
            'emotion_present': False
        },
        {
            'path': os.path.join(raw_data, 'Helen_Taylor'),
            'speaker_id': 6,  # 9136,
            'process_audio': True,
            'emotion_present': False
        },
        {
            'path': os.path.join(raw_data, 'Sylviamb'),
            'speaker_id': 7,  # 11614,
            'process_audio': True,
            'emotion_present': False
        },
        {
            'path': os.path.join(raw_data, 'Celine_Major'),
            'speaker_id': 8,  # 11697,
            'process_audio': True,
            'emotion_present': False
        },
        {
            'path': os.path.join(raw_data, 'LikeManyWaters'),
            'speaker_id': 9,  # 12787,
            'process_audio': True,
            'emotion_present': False
        }
    ]

    emo_id_map = {
        'neutral-normal': 0,
        'calm-normal': 1,
        'calm-strong': 2,
        'happy-normal': 3,
        'happy-strong': 4,
        'sad-normal': 5,
        'sad-strong': 6,
        'angry-normal': 7,
        'angry-strong': 8,
        'fearful-normal': 9,
        'fearful-strong': 10,
        'disgust-normal': 11,
        'disgust-strong': 12,
        'surprised-normal': 13,
        'surprised-strong': 14
    }
