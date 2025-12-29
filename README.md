# xlsr-english-l2
This repo is used to fintune facebook/wav2vec2-xlsr-53-espeak-cv-ft to transcript the phonemes of L2 english speacker

## 0. Setup enviroment and packages
- Download [uv](https://docs.astral.sh/uv/getting-started/installation/) for managing 
- Install all packages (run terminal with local enviroment):
```cmd
uv sync
```

## 1. Process data
- Download dataset from [L2-arctic](https://psi.engr.tamu.edu/l2-arctic-corpus/) to folder [dataset](./dataset/) and extract all
- 
## 2. Finetune model [wav2vec2-xlsr-53-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft) model for Mispronounciation Detection and Diagnosis
- 