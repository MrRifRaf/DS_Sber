import os

import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import wavfile

from router import models
from tacotron2.text import text_to_sequence

matplotlib.use('Agg')

RATE = 22050
CURR_PATH = os.path.dirname(os.path.abspath(__file__))
HAS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if HAS_CUDA else 'cpu'


def main():
    t2, wg = load_models()
    text = input('Please input some phrase: ')
    # text = 'Hello, how are you today?'
    speaker_id = int(input('Input speaker id: '))
    # speaker_id = 4
    infer(t2, wg, text, speaker_id)


def load_models():
    wg_path = os.path.join(CURR_PATH, 'mnt', 'pretrained', 'wg_fp32_torch')
    taco_path = os.path.join(CURR_PATH, 'mnt', 'pretrained', 'checkpoint_last')

    taco_checkpoint = torch.load(taco_path, map_location=DEVICE)
    wg_checkpoint = torch.load(wg_path, map_location=DEVICE)

    t2 = models.get_model('Tacotron2',
                          taco_checkpoint['config'],
                          to_cuda=HAS_CUDA)
    wg = models.get_model('WaveGlow',
                          wg_checkpoint['config'],
                          to_cuda=HAS_CUDA)

    for model, checkpoint in [(t2, taco_checkpoint), (wg, wg_checkpoint)]:
        new_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)

    t2.eval()
    wg.eval()
    return t2, wg


def infer(t2, wg, text, speaker_id=0):
    infer_path = os.path.join(CURR_PATH, 'static', 'infer')
    inputs = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    inputs = torch.from_numpy(inputs).to(device=DEVICE, dtype=torch.int64)
    speaker_id = torch.IntTensor([speaker_id]).long()

    with torch.inference_mode():
        _, mel, _, _ = t2.infer(inputs, speaker_id, on_cuda=HAS_CUDA)
        audio = wg.infer(mel)
    # plt.imshow(mel.squeeze(0).detach().cpu().numpy(), aspect='auto')
    mel = mel.squeeze(0).detach().cpu().numpy()
    img = librosa.display.specshow(mel,
                                   y_axis='log',
                                   sr=RATE,
                                   hop_length=256,
                                   x_axis='time')
    plt.colorbar(img, format="%+2.f dB")
    plt.savefig(os.path.join(infer_path, 'mel.png'))
    plt.clf()

    audio_numpy = audio[0].data.cpu().numpy()

    wavfile.write(os.path.join(infer_path, 'result.wav'), RATE, audio_numpy)


if __name__ == '__main__':
    main()
