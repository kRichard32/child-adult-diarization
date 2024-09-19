import argparse

from models.whisper import WhisperWrapper
import torch
import numpy as np
from subprocess import CalledProcessError, run

SAMPLE_RATE = 16000


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform diarization on an audio sample and print '
                                                 'a tensor relating to the results')

    parser.add_argument('-i', '--audio', type=str, required=True, help='The input audio')
    args = parser.parse_args()

    audio = args.audio

    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    audio = audio.unsqueeze(0)
    model = WhisperWrapper()
    model.backbone_model.encoder.embed_positions = model.backbone_model.encoder.embed_positions.from_pretrained(
        model.embed_positions[:500])
    model.load_state_dict(torch.load("whisper-base_rank8_pretrained_50k.pt"))
    model.cuda()
    output = model.forward_eval(audio)
    torch.set_printoptions(threshold=10_000)

    print(output)
    tensor_array = output.detach().numpy()
    time = 0
    diarization_dictionary = {}
    for i in range(len(tensor_array)):
        mapping = {0: "silence", 1: "child", 2: "adult", 3: "overlap"}
        diarization_dictionary[time/1000] = mapping[tensor_array[i]]
        time += 20
    print(diarization_dictionary)

