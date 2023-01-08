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

from tacotron2.text import symbols


def tacotron2_parser(args, _):
    """ Add additional hyperparameters related to tacatron2 model.
    """

    # misc parameters
    args.mask_padding = False  # Use mask padding
    args.n_mel_channels = 80  # Number of bins in mel-spectrograms

    # symbols parameters
    len_symbols = len(symbols)
    args.n_symbols = len_symbols  # Number of symbols in dictionary
    args.symbols_embedding_dim = 512 // 2  # Input embedding dimension

    # encoder parameters
    args.encoder_kernel_size = 5  # Encoder kernel size
    args.encoder_n_convolutions = 3  # Number of encoder convolutions
    args.encoder_embedding_dim = 512 // 2  # Encoder embedding dimension

    # decoder parameters

    # Number of frames processed per step. Currently only 1 is supported
    args.n_frames_per_step = 1

    args.decoder_rnn_dim = 1024  # Number of units in decoder LSTM

    # Number of ReLU units in prenet layers
    args.prenet_dim = 256

    # Maximum number of output mel spectrograms
    args.max_decoder_steps = 2000

    args.gate_threshold = 0.5  # Probability threshold for stop token
    args.p_attention_dropout = 0.1  # Dropout probability for attention LSTM
    args.p_decoder_dropout = 0.1  # Dropout probability for decoder LSTM

    # Stop decoding once all samples are finished
    args.decoder_no_early_stopping = False

    # attention parameters
    args.attention_rnn_dim = 1024  # Number of units in attention LSTM

    # Dimension of attention hidden representation
    args.attention_dim = 128

    # location layer parameters

    # Number of filters for location-sensitive attention
    args.attention_location_n_filters = 32

    # Kernel size for location-sensitive attention
    args.attention_location_kernel_size = 31

    # Mel-post processing network parameters
    args.postnet_embedding_dim = 512  # Postnet embedding dimension
    args.postnet_kernel_size = 5  # Postnet kernel size
    args.postnet_n_convolutions = 5  # Number of postnet convolutions

    args.n_speakers = 2
    args.speakers_embedding_dim = 2
    return args
