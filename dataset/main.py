import csv
import datetime
import glob
import os
import sys
from concurrent import futures
from typing import Callable, Dict

import torch
from torch.utils.data import TensorDataset

from src.dataset.similarity import *
from src.dataset.domain.label import Label


@timed_func
def load_audio(path: str, duration: int, offset: int, limit: int = sys.maxsize, sample_ratio: float = 1) \
        -> frozenset[Track]:
    """
    Recursively reads .wav files from the root directory specified into main memory.

    :param path: root directory to read audio files from
    :param duration: how long to read from each file; specified in seconds
    :param offset: position in file where the reading starts; specified in seconds
    :param limit: limits the number of files to read
    :param sample_ratio: specifies sampling ratio for uniform random sample draw
    :return: frozenset containing Track objects
    """
    tracks: List[Track] = []

    filenames = list(glob.iglob(path + '**/*.wav', recursive=True))
    np.random.seed(seed=69420)
    random_filenames = np.random.choice(filenames, int(len(filenames) * sample_ratio))

    for i, filename in enumerate(random_filenames):
        track = Track.from_fp(os.path.join(path, filename), duration, offset)
        if track.metadata.duration >= duration:
            tracks.append(track)
        else:
            print(f'Omitting file: {track.metadata.filename} as it has shorter duration: {track.metadata.duration}'
                  f' as specified: {duration}')
        if i == limit:
            return frozenset(tracks)

    if len(tracks) == 0:
        raise RuntimeError(f'No tracks were loaded from the provided directory: {path}')

    print(f'Loaded {len(tracks)} audio files')

    return frozenset(tracks)


def __similarity_score_pipeline(steps: List[Callable[[frozenset[Track]], Results]],
                                tracks: frozenset[Track],
                                weights: Dict[str, float] = None) -> np.ndarray:
    if weights is not None and (len(weights) != len(steps)):
        raise ValueError(
            f'Number of weights doesn\'t match the number of number of functions in the pipeline.'
            f' Got {len(weights)}, expected: {len(steps)}'
        )
    if weights is not None and (round(sum(weights.values()))) != 1:
        raise ValueError(f'Incorrect weights. Expected weights sum equal to 1, but got: {sum(weights.values())}.')

    workers = min(os.cpu_count(), len(steps))
    queue = []

    with futures.ProcessPoolExecutor(workers) as executor:
        for func in steps:
            queue.append(executor.submit(func, tracks))

    results: List[Results] = [job.result() for job in queue]
    similarity_scores: List[List[float]] = [result.values for result in results]
    weights_vec = [weights[result.similarity_func] for result in results] if weights \
        else [1.0 / len(steps) for _ in range(len(steps))]

    return np.average(similarity_scores, axis=0, weights=weights_vec)


def __feature_computation_pipeline(tracks: frozenset[Track],
                                   feature_extractor: Callable[[List[Track]], List[np.ndarray]]) -> List[np.ndarray]:
    n_workers = min(os.cpu_count(), len(tracks))
    queue = []
    tracks = list(tracks)  # frozenset is not subscriptable cast to list
    chunks = [tracks[i:i + n_workers] for i in list(range(0, len(tracks), n_workers))]

    with futures.ProcessPoolExecutor(n_workers) as executor:
        for _chunk in chunks:
            queue.append(executor.submit(feature_extractor, _chunk))

    results = [job.result() for job in queue]
    results = [x for features in results for x in features]

    assert len(results) == len(tracks)

    return results


@timed_func
def __join_features_and_labels(features: List[np.ndarray], labels: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = [[i, j] for i, j in combinations(range(len(features)), 2)]
    if len(indices) != labels.size:
        raise ValueError(
            f'The number of labels: {labels.size} doesn\'t match the number of feature pairs: {len(indices)}')
    feature_pairs = torch.tensor(np.array([[features[i], features[j]] for i, j in indices]))

    return feature_pairs, torch.tensor(labels)


def __build_track_similarity_map(tracks: frozenset[Track], labels: np.ndarray):
    tracks_similarity_map: Dict[float, List[str]] = {}

    for score, (track_a, track_b) in zip(labels, combinations(tracks, 2)):
        remove_file_extension = lambda filename: filename.replace('.wav', '')
        tracks_similarity_map[score] = [remove_file_extension(track_a.metadata.filename),
                                        remove_file_extension(track_b.metadata.filename)]

    return tracks_similarity_map


def generate_and_save_similarity_map(tracks: frozenset[Track], labels: np.ndarray, output_dir: str) -> None:
    """ Generates and saves a map of (similarity score, track_a.filename, track_b.filename)
        Can be useful for creating an insight into the score distribution, and data quality assurance

        :param tracks: frozenset of Track objects
        :param labels: a numpy array containing scores
        :param output_dir: the directory in which the dataset will get stored after a successful computation

        Notes:
            The labels must be obtained from similarity_score_pipeline without changing their order
    """
    output_filename = f'scores-{np.random.randint(69420)}'

    track_similarity_map = __build_track_similarity_map(tracks, labels)

    with open(f'{output_dir}/{output_filename}.csv', 'w') as f:
        w = csv.writer(f)
        for score, titles in track_similarity_map.items():
            w.writerow([score, titles[0], titles[1]])

    print(f'Files saved as: {output_filename}')


def generate_and_save_dataset(tracks: frozenset[Track],
                              feature_extractor: Callable[[List[Track]], List[np.ndarray]],
                              similarity_score_pipeline_funcs: List[Callable[[frozenset[Track]], Results]],
                              output_dir: str, weights: Dict[str, float] = None) -> None:
    """
    Generates and saves a torch dataset containing pairs of features and the corresponding scores.
    The resulting dataset has the following shape (n, 2, 1) where n is the number of instances

    :param tracks: frozenset containing a list of Track objects
    :param similarity_score_pipeline_funcs: a list that contains similarity computation functions,
            the functions are applied on sequentially, but each step is executed in parallel (SIMD)
    :param output_dir: the directory in which the dataset will get stored after a successful computation
    :param weights: Optional parameter to specify weights for each sequential similarity function;
            may be helpful to assign higher importance to certain similarity functions
    :return:
    """

    features = __feature_computation_pipeline(tracks, feature_extractor)
    scores = __similarity_score_pipeline(similarity_score_pipeline_funcs, tracks, weights)

    feature_pairs, scores = __join_features_and_labels(features, scores)
    labels = __assign_labels(scores)
    feature_pairs_test, feature_pairs_train, labels_test, labels_train = __get_train_test_split(feature_pairs, labels)

    torch.save(TensorDataset(feature_pairs_train, labels_train),
               f'{output_dir}/pairs_train-{datetime.datetime.now()}.pt')
    torch.save(TensorDataset(feature_pairs_test, labels_test), f'{output_dir}/pairs_test-{datetime.datetime.now()}.pt')
    generate_and_save_similarity_map(tracks, scores.detach().cpu().numpy(), output_dir)


def generate_and_save_unsupervised_dataset(tracks: frozenset[Track],
                                           feature_extractor: Callable[[List[Track]], List[np.ndarray]],
                                           similarity_score_pipeline_funcs: List[Callable[[frozenset[Track]], Results]],
                                           output_dir: str, weights: Dict[str, float] = None) -> None:
    """
    Generates a torch dataset for unsupervised learning that contains features only (no labels).
    The train data consists of vectors only, while the test data consists of vector pairs and the corresponding
     similarity class.

    :param tracks:
    :param feature_extractor:
    :param output_dir:
    :return:
    """

    n_train_features = int(0.7 * len(tracks))
    features = __feature_computation_pipeline(tracks, feature_extractor)
    features_train, features_test = features[:n_train_features], features[n_train_features:]
    tracks_test = frozenset(list(tracks)[n_train_features:])
    scores = __similarity_score_pipeline(similarity_score_pipeline_funcs, tracks_test, weights)
    feature_pairs_test, scores = __join_features_and_labels(features_test, scores)
    labels = __assign_labels(scores)

    torch.save(TensorDataset(torch.tensor(np.array(features_train))),
               f'{output_dir}/unsupervised_train-{datetime.datetime.now()}.pt')
    torch.save(TensorDataset(feature_pairs_test, labels),
               f'{output_dir}/unsupervised_test-{datetime.datetime.now()}.pt')


def __assign_labels(scores: torch.Tensor) -> torch.Tensor:
    """
     Given similarity scores, assign labels for classification. The labels are defined in an enum class.
     The similarity scores array will be sorted and divided in n partitions, where n is specified by the number of label
     in the enum. Then each of the partitions will get labels assigned.
     This way the distribution of labels is guaranteed to be uniform.

     Example:
     scores = [0.53109224 0.35676397 0.51311614 0.32530468 0.45724967]
     labels = [0. 1. 3. 0. 2.]

    :param scores: np.ndarray containing scores as float numbers
    :return: np.ndarray that contains labels representing similarity as defined in the enum class.
    """
    sorted_indices = torch.argsort(scores)
    n_splits = len(Label)
    n_scores = len(scores)
    start, step = 0, round(n_scores / n_splits)
    labels = torch.zeros(n_scores)

    for n in range(n_splits):
        labels[sorted_indices[start:start + step]] = n
        start += step

    labels[-step:1] = n_splits
    return labels.long()


def __get_train_test_split(feature_pairs: torch.Tensor, labels: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_feature_pairs = feature_pairs.shape[0]
    n_test_samples = int(0.3 * n_feature_pairs)
    split_size = [n_feature_pairs - n_test_samples, n_test_samples]
    feature_pairs_train, feature_pairs_test = feature_pairs.split(split_size)
    labels_train, labels_test = labels.split(split_size)
    return feature_pairs_test, feature_pairs_train, labels_test, labels_train


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, '../../data/raw/16000/')
    OUTPUT_DIR = os.path.join(ROOT_DIR, '../../data/')
    DURATION = 20  # only load up to this much audio (in seconds)
    OFFSET = 30  # start reading after this time (in seconds)

    tracks: frozenset[Track] = load_audio(path=DATA_DIR, duration=DURATION, offset=OFFSET, sample_ratio=.5)

    pipeline: List[Callable[[frozenset[Track]], Results]] = [
        mfcc_similarity,
        #mel_spec_similarity,
        # zcr_similarity,
        # chroma_similarity,
        # rhythm_similarity,
        #spectral_contrast_similarity,
        # perceptual_similarity, # Takes long
    ]

    weights: Dict[str, float] = {
        mfcc_similarity.__name__: 1.0,
        #mel_spec_similarity.__name__: 1.0,
        #zcr_similarity.__name__: .10,
        #chroma_similarity.__name__: .15,
        #rhythm_similarity.__name__: .15,
        #spectral_contrast_similarity.__name__: .10
    }

    generate_and_save_dataset(tracks, mfcc_dataset, pipeline, OUTPUT_DIR, weights)

    #generate_and_save_unsupervised_dataset(tracks, vggish_embedding, pipeline, OUTPUT_DIR, weights)
