import logging
import os
from dataclasses import dataclass
from typing import Tuple

import librosa
import numpy as np

logger = logging.getLogger()


@dataclass
class TrackMetadata:
    filename: str
    sampling_rate: float
    duration: int


class Track:
    def __init__(self, metadata: TrackMetadata, audio: np.ndarray):
        self.metadata: TrackMetadata = metadata
        self.audio: np.ndarray = audio

    @classmethod
    def from_fp(cls, fp: str, duration: int, offset: int):
        metadata, audio = cls._load_from_fp(fp, duration, offset)
        return cls(metadata, audio)

    @classmethod
    def from_ndarray(cls, audio: np.ndarray, sr: int):
        return cls(TrackMetadata(filename=f'file-{np.random.randint(69420)}',
                                 sampling_rate=sr,
                                 duration=audio.size // sr), audio)

    @staticmethod
    def _load_from_fp(fp: str, duration: int, offset: int) -> Tuple[TrackMetadata, np.ndarray]:
        try:
            sample_rate = librosa.get_samplerate(fp)
            audio, _ = librosa.load(f'{fp}',
                                    duration=duration,
                                    sr=sample_rate,
                                    offset=offset)
            metadata = TrackMetadata(
                filename=os.path.basename(fp),
                sampling_rate=sample_rate,
                duration=int(librosa.get_duration(y=audio, sr=sample_rate))
            )
            return metadata, audio
        except FileNotFoundError as e:
            logging.error(f"Error loading file {fp}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error loading file {fp}: {type(e).__name__}, {e}")

    def __eq__(self, other):
        return self.metadata.filename == other.metadata.filename

    def __hash__(self):
        return hash(self.metadata.filename)
