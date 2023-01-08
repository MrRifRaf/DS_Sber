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


def waveglow_parser(args, _):
    """ Add additional hyperparameters related to WaveGlow model.
    """

    # misc parameters
    args.n_mel_channels = 80  # Number of bins in mel-spectrograms

    # glow parameters

    args.flows = 12  # Number of steps of flow

    # Number of samples in a group processed by the steps of flow
    args.groups = 8

    # Determines how often (i.e., after how many coupling layers)
    # a number of channels (defined by --early-size parameter) are output
    # to the loss function
    args.early_every = 4

    args.early_size = 2  # Number of channels output to the loss function
    args.sigma = 1.0  # Standard deviation used for sampling from Gaussian

    # Segment length (audio samples) processed per iteration
    args.segment_length = 4000

    # wavenet parameters

    # Kernel size for dialted convolution in the affine coupling layer (WN)
    args.wn_kernel_size = 3

    args.wn_channels = 512 // 2  # Number of channels in WN
    args.wn_layers = 8  # Number of layers in WN

    return args
