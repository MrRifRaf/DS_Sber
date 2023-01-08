import matplotlib.pyplot as plt
import torch
from scipy.io.wavfile import write


class Model:

    def __init__(self):

        self.tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub',
                                        'nvidia_tacotron2',
                                        pretrained=False)

        checkpoint = torch.hub.load_state_dict_from_url(
            'https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2pyt_fp32/versions/1/files/nvidia_tacotron2pyt_fp32_20190306.pth',
            map_location="cpu")

        # Unwrap the DistributedDataParallel module
        # module.layer -> layer
        state_dict = {
            key.replace("module.", ""): value
            for key, value in checkpoint["state_dict"].items()
        }

        # Apply the state dict to the model
        self.tacotron2.load_state_dict(state_dict)
        self.tacotron2.eval()

        self.waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                                       'nvidia_waveglow',
                                       pretrained=False)
        checkpoint = torch.hub.load_state_dict_from_url(
            'https://api.ngc.nvidia.com/v2/models/nvidia/waveglowpyt_fp32/versions/1/files/nvidia_waveglowpyt_fp32_20190306.pth',
            map_location="cpu")

        # Unwrap the DistributedDataParallel module
        # module.layer -> layer
        state_dict = {
            key.replace("module.", ""): value
            for key, value in checkpoint["state_dict"].items()
        }

        # Apply the state dict to the model
        self.waveglow.load_state_dict(state_dict)
        self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
        self.waveglow.eval()
        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                                    'nvidia_tts_utils')

    def text2speech(self, text):
        sequences, lengths = self.utils.prepare_input_sequence([text],
                                                               cpu_run=True)

        with torch.inference_mode():
            mel, _, alignment = self.tacotron2.infer(sequences, lengths)
            audio = self.waveglow.infer(mel)
        audio_numpy = audio[0].data.cpu().numpy()
        rate = 22050
        align_path = "alignment.png"
        plt.imshow(alignment.float().data.cpu().numpy().T,
                   aspect="auto",
                   origin="lower")
        plt.savefig(align_path)

        write("audio.wav", rate, audio_numpy)


def main():
    text = "I don't want to put away the suitcases"
    model = Model()
    model.text2speech(text)


if __name__ == '__main__':
    main()
