# Audio Similarity Experiment for Master Thesis dyploma

Thesis title: "Application of siamese naturalne networks to evaluate similarity of musical tracks"

Collection of tools for experimenting with notion of similarity of audio files.
Contains modules for generating audio datasets, extracting audio features and computing similarity scores.

## Development
1. Create virtual environment and install dependencies:
```shell
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
1. Put audio files under `data/raw/`
1. (Optional) transcode to PCM 16kHz with `transcodemp3topcm.sh`. Librosa does transcoding itself on every run otherwise.
1.  Generate the dataset 
```shell
cd src/dataset && python main.py
```
