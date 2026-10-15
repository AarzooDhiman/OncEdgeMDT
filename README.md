# OncEdge MDT

Research code accompanying a manuscript on automated transcription and
retrieval-augmented processing of multidisciplinary team (MDT) meeting material.
The associated audio dataset is not included in this GitHub upload.

## Repository contents

| Path | Contents |
| --- | --- |
| `transcription/` | Scripts and non-AWS notebooks for Whisper, Parakeet, and WhisperX workflows. The AWS Transcribe notebook is excluded from this upload. |
| `optimal_transcription/` | A Whisper workflow with voice activity detection and audio-quality processing. |
| `optuna/` | Whisper optimization scripts and job files. |
| `data/` (not included) | The associated audio dataset, augmentation materials, and tiered case files. |
| `RAG/` | Code for chunking, retrieval, and language-model workflows, plus a merged JSON output. |
| `RAG/NICE_guidelines/` | Four NICE guideline PDF documents used by the RAG workflows. |

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

## Reproducing the workflows

No unified dependency lockfile or environment specification is included.
Python package requirements and external services vary by script and notebook.
Review the relevant file before running it, and configure any required
credentials and model/service access through your local environment. Do not
commit credentials or other secrets.

## Data and access

The source project includes meeting audio, scripts, derived audio, and clinical
guideline documents; the audio dataset is not part of this GitHub repository.
Access and reuse should follow the applicable research approvals, data-sharing
agreements, and document licensing terms. Keep the repository private unless
all materials are explicitly cleared for wider distribution.

## GitHub large-file note

The omitted audio dataset is too large for a practical regular Git history, and
two files exceed GitHub's 100 MiB per-file limit. Git Large File Storage
(Git LFS) or an approved external data host is needed before distributing the
full dataset. LFS storage and bandwidth quotas may apply.
