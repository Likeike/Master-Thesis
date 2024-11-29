from typing import List

import essentia.standard as es
import librosa
import numpy as np
from os import path

from src.dataset.domain import track
from src.dataset.domain.track import Track
def mfcc(track: Track) -> np.ndarray:
    mel_spectrogram = librosa.feature.melspectrogram(track.audio, sr=track.metadata.sampling_rate,
                                                     win_length = 1024,
                                                     hop_length = 128,
                                                     window = 'hann',
                                                     n_fft = 1024,
                                                     n_mels = 40)

    return librosa.feature.mfcc(S = librosa.power_to_db(mel_spectrogram, ref=np.max),
                                sr=track.metadata.sampling_rate,
                                n_mfcc = 13)
def mfcc_dataset(tracks: List[Track]) -> List[np.ndarray]:
    return [librosa.feature.mfcc(S=librosa.amplitude_to_db(librosa.feature.melspectrogram(track.audio,
                                                                                          sr=track.metadata.sampling_rate,
                                                                                          n_mels=40,
                                                                                          win_length=1024,
                                                                                          hop_length=128,
                                                                                          window='hann',
                                                                                          n_fft=1024), ref=np.max),
                                 sr=track.metadata.sampling_rate,
                                 n_mfcc = 13,
                                 ) for track in tracks]

def mel_spec(tracks: List[Track]) -> List[np.ndarray]:
    return [librosa.feature.melspectrogram(y=librosa.amplitude_to_db(track.audio, ref=np.max),
                                           sr=track.metadata.sampling_rate,
                                           win_length=1024,
                                           n_fft=1024,
                                           hop_length=128,
                                           window='hann',
                                           n_mels=40
                                          ) for track in tracks]

def zcr(track: Track) -> float:
    return np.mean(np.abs(np.diff(np.sign(track.audio))) > 0)


def vggish_embedding(tracks: List[Track]) -> List[np.ndarray]:
    # requires pre-trained VGGish model
    # wget -q https://essentia.upf.edu/models/feature-extractors/vggish/audioset-vggish-3.pb
    model = es.TensorflowPredictVGGish(
        graphFilename=path.join(path.abspath(path.curdir), 'dataset/audioset-vggish-3.pb'),
        output='model/vggish/embeddings')
    return [np.average(model(track.audio), axis=0) for track in tracks]


def onset_vector(track: Track) -> np.ndarray:
    min_length = len(track.audio)
    onset_vector = np.zeros(min_length)
    onsets = librosa.onset.onset_detect(y=track.audio, sr=track.metadata.sampling_rate, units='time')
    onsets = np.array(onsets) * track.metadata.sampling_rate
    onsets = onsets[onsets < min_length]
    onset_vector[onsets.astype(int)] = 1
    return onset_vector


def chromagram(track: Track) -> np.ndarray:
    return librosa.feature.chroma_stft(y=track.audio, sr=track.metadata.sampling_rate)


def spectral_contrast(track: Track) -> np.ndarray:
    return librosa.feature.spectral_contrast(y=track.audio, sr=track.metadata.sampling_rate)
