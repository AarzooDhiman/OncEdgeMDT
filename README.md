# OncEdge MDT

OncEdge MDT is a fully on-device Automatic Speech Recognition and  Retrieval-Augmented Generation (RAG)-based Large Language Model (LLM) pipeline to streamline Multidisciplary Team Meetings for Breast Cancer Oncology. 

## Repository contents

| Path | Contents |
| --- | --- |
| `transcription/` | Scripts and non-AWS notebooks for Whisper, Parakeet, and WhisperX workflows. |
| `optimal_transcription/` | A optimised Whisper workflow with silence-aware preprocessing and noise-adaptive decoding as preprocessing and postprocessing, reaching performance comparable to the closed-source models |
| `optuna/` | Whisper optimization scripts and job files. |
| `data/` | The associated audio dataset, augmentation materials, and tiered case files. |
| `RAG/NICE_guidelines/` | Four NICE guideline PDF documents used by the RAG workflows. |
| `RAG/` | Code for chunking, chunked files, retrieval, and language-model workflows that uses context from NICE guidelines. |


The excluded audio collection is approximately 5.1 GB. Its `data/tier3/`
directory contains 127 audio files for each of ten cases.

## Workflows

The folders separate the main research activities:

1. Transcribe meeting recordings using the selected scripts or notebooks in
   `transcription/`.
2. Explore Whisper parameter optimization in `optuna/` and audio processing in
   `optimal_transcription/`.
3. Explore guideline chunking, retrieval, and model workflows in `RAG/`.

These files are research workflows rather than a single packaged application;
the appropriate entry point and inputs depend on the experiment being run.
Audio augmentation code and inputs under `data/` are not included in this upload.

## Dataset
The data attached to this work is available at [Data Repo](https://figshare.com/s/27501bb3e1096f720c55)



