from dataclasses import dataclass
from itertools import combinations
from typing import Tuple, Set

import pystoi
from dtw import accelerated_dtw

from src.dataset.features import *
from src.dataset.utils import timed_func


@dataclass
class Results:
    values: List[float]
    similarity_func: str


@timed_func
def zcr_similarity(tracks: Set[Track]) -> Results:
    """
    Calculate the zero-crossing rate (ZCR) similarity between audio files.

    Returns:
        float: The average ZCR similarity score between the audio files, normalized between 0 and 1.

    Raises:
        None

    Notes:
        The ZCR similarity is calculated by comparing the zero-crossing rates of the audio signals.
        Zero-crossing rate represents the rate at which the audio signal changes its sign. The similarity
        score is obtained by calculating the absolute difference between the ZCR of the original and compare
        audio files and subtracting it from 1. The similarity score ranges between 0 and 1, where a higher
        score indicates greater similarity.

    """

    features = [zcr(track) for track in tracks]
    return Results(
        values=[_compute_zcr_similarity((features[i], features[j])) for i, j in combinations(range(len(features)), 2)],
        similarity_func=zcr_similarity.__name__
    )


@timed_func
def mfcc_similarity(tracks: frozenset[Track]) -> Results:
    features = [mfcc(track) for track in tracks]
    similarities = [_compute_mfcc_similarity((features[i], features[j])) for i, j in
                    combinations(range(len(features)), 2)]
    return Results(
        values=similarities,
        similarity_func=mfcc_similarity.__name__
    )

@timed_func
def mel_spec_similarity(tracks: frozenset[Track]) -> Results:
    features = [mel_spec(track) for track in tracks]
    similarities = [_compute_mel_spec_similarity((features[i], features[j]))for i,j in
                    combinations(range(len(features)), 2)]
    return Results(
        values=similarities,
        similarity_func=mel_spec_similarity.__name__
    )

@timed_func
def perceptual_similarity(tracks: Set[Track]) -> Results:
    """
    Calculate the perceptual similarity between audio files using the Short-Time Objective Intelligibility (STOI) metric.

    :param tracks:

    Returns:
        float: The average perceptual similarity score between the audio files, normalized between 0 and 1.

    Raises:
        Assertion Error

    Notes:
        The perceptual similarity is calculated using the Short-Time Objective Intelligibility (STOI) metric.
        STOI measures the similarity of two audio signals in terms of their intelligibility. The STOI score
        ranges between -1 and 1, where a higher score indicates greater similarity. The perceptual similarity
        score is obtained by normalizing the STOI score between 0 and 1, where 0 indicates no similarity and 1
        indicates perfect similarity.

    """
    features = list(tracks)
    return Results(
        values=[_compute_perceptual_similarity((features[i], features[j])) for i, j in
                combinations(range(len(features)), 2)],
        similarity_func=perceptual_similarity.__name__
    )


@timed_func
def spectral_contrast_similarity(tracks: Set[Track]) -> Results:
    features = [spectral_contrast(track) for track in tracks]
    return Results(
        values=[_compute_spectral_contrast_similarity((features[i], features[j])) for i, j in
                combinations(range(len(features)), 2)],
        similarity_func=spectral_contrast_similarity.__name__
    )


@timed_func
def rhythm_similarity(tracks: Set[Track]) -> Results:
    features = [onset_vector(track) for track in tracks]
    return Results(
        values=[_compute_rhythm_similarity((features[i], features[j])) for i, j in
                combinations(range(len(features)), 2)],
        similarity_func=rhythm_similarity.__name__
    )


@timed_func
def chroma_similarity(tracks: Set[Track]) -> Results:
    features = [chromagram(track) for track in tracks]
    return Results(
        values=[_compute_chroma_similarity((features[i], features[j])) for i, j in
                combinations(range(len(features)), 2)],
        similarity_func=chroma_similarity.__name__
    )


def _compute_rhythm_similarity(pair: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Calculate the rhythm similarity between audio files.

    Returns:
        float: The average rhythm similarity score between the audio files, normalized between 0 and 1.

    Raises:
        AssertionError

    Notes:
        The rhythm similarity is calculated by comparing the rhythm patterns of the audio signals.
        Rhythm patterns are derived from the onsets in the audio. The similarity score is obtained
        by calculating the Pearson correlation coefficient between the rhythm patterns of the original
        and compare audio files and normalizing it between 0 and 1. The similarity score ranges between
        0 and 1, where a higher score indicates greater similarity.

    """
    onset_vector_a, onset_vector_b = pair
    min_length = min(len(onset_vector_a), len(onset_vector_b))
    similarity = (np.corrcoef(onset_vector_a[:min_length], onset_vector_b[:min_length])[0, 1] + 1) / 2
    assert 0 <= similarity <= 1

    return similarity


def _compute_chroma_similarity(pair: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Calculate the chroma similarity between audio files.

    Returns:
        float: The average chroma similarity score between the audio files, normalized between 0 and 1.

    Raises:
        AssertionError

    Notes:
        The chroma similarity is calculated by comparing the chroma features of the audio signals.
        Chroma features represent the distribution of pitches in the audio. The similarity score is
        obtained by calculating the mean absolute difference between the chroma features of the original
        and compare audio files, and subtracting it from 1. The similarity score ranges between 0 and 1,
        where a higher score indicates greater similarity.
    """
    chroma_a, chroma_b = pair
    min_length = min(chroma_a.shape[1], chroma_b.shape[1])
    chroma_a = chroma_a[:, :min_length]
    chroma_b = chroma_b[:, :min_length]
    similarity = 1 - np.mean(np.abs(chroma_a - chroma_b))
    assert 0 <= similarity <= 1

    return similarity


def _compute_spectral_contrast_similarity(pair: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Calculate the spectral contrast similarity between audio files.

    Returns:
        float: The average spectral contrast similarity score between the audio files, normalized between 0 and 1.

    Raises:
        AssertionError

    Notes:
        The spectral contrast similarity is calculated by comparing the spectral contrast of the audio signals.
        Spectral contrast measures the difference in magnitudes between peaks and valleys in the spectrum,
        representing the perceived amount of spectral emphasis. The spectral contrast similarity score is
        obtained by comparing the spectral contrast of the original and compare audio files and calculating
        the average normalized similarity. The similarity score ranges between 0 and 1, where a higher score
        indicates greater similarity.
    """
    spectral_contrast_a, spectral_contrast_b = pair
    min_columns = min(spectral_contrast_a.shape[1], spectral_contrast_b.shape[1])
    spectral_contrast_a = spectral_contrast_a[:, :min_columns]
    spectral_contrast_b = spectral_contrast_b[:, :min_columns]
    similarity = np.mean(np.abs(spectral_contrast_a - spectral_contrast_b))
    normalized_similarity = (1 - similarity / np.max(
        [np.abs(spectral_contrast_a), np.abs(spectral_contrast_b)]))
    assert 0 <= normalized_similarity <= 1

    return normalized_similarity


def _compute_perceptual_similarity(pair: Tuple[Track, Track]) -> float:
    track_a, track_b = pair
    min_length = min(len(track_a.audio), len(track_b.audio))
    similarity = (pystoi.stoi(track_a.audio[:min_length], track_b.audio[:min_length],
                              min(track_b.metadata.sampling_rate, track_b.metadata.sampling_rate)) + 1) / 2

    return similarity


def _compute_mfcc_similarity(pair: Tuple[np.ndarray, np.ndarray]) -> float:
    d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(pair[0], pair[1], dist='euclidean', warp=1)
    similarity = 1 / (1 + d)

    assert 0 <= similarity <= 1

    return similarity


def _compute_zcr_similarity(pair: Tuple[float, float]) -> float:
    similarity = 1 - np.abs(pair[0] - pair[1]) / 2

    return similarity

def _compute_mel_spec_similarity(pair: Tuple[np.ndarray, np.ndarray]) -> float:
    d, cont_matrix, acc_cost_matrix, path = accelerated_dtw(pair[0], pair[1], dist='euclidean', warp=1)
    similarity = 1 / (1 + d)

    assert 0 <= similarity <= 1

    return similarity
