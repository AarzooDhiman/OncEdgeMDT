# OncEdge MDT

**A fully on-device Automatic Speech Recognition (ASR) and Retrieval-Augmented Generation (RAG) pipeline for Breast Cancer Oncology Multidisciplinary Team (MDT) meetings.**

© University of Hull. All rights reserved. This repository is proprietary; see [License](#license--contribution-guidelines).

> **Data availability.** The `data/` folder (audio, reference transcripts, augmentation code and noise clips) is **not stored in this GitHub repository** because of its size (approx. 5.1 GB). It is available through Figshare **upon request**; please contact the corresponding author (a.dhiman@hull.ac.uk). Once obtained, extract it into a `data/` folder at the root of this repository before running the code (see [Dataset Information](#dataset-information)).

## Description

OncEdge MDT supports the transcription and summarisation of Multidisciplinary Team meetings without sending patient-related audio or text to external services. The repository contains the research code and dataset accompanying the associated journal manuscript. It covers:

1. **Transcription** of MDT meeting audio with several ASR systems (Whisper, WhisperX, NVIDIA Parakeet, AWS Transcribe).
2. **Optimisation** of Whisper decoding parameters with Optuna.
3. **An optimised Whisper workflow** using silence-aware (VAD) preprocessing and noise-adaptive decoding, reaching performance comparable to closed-source models.
4. **A RAG workflow** that chunks NICE breast cancer guidelines, builds a vector store, retrieves relevant guidance and passes it, with the case transcript, to local medical LLMs.

These are research workflows, not a single packaged application. The entry point depends on the experiment being run.

## Dataset Information

The audio dataset (approx. 5.1 GB) is organised in tiers of increasing acoustic difficulty. It consists of ten simulated breast cancer MDT cases (`case1` to `case10`) plus two short simulated MDT discussions in `tier1`. No real patient cases or data are used.

**Where to get it.** The data is available through Figshare upon request (contact a.dhiman@hull.ac.uk). It is not included in this GitHub repository. Once obtained, extract it into the `data/` folder of this repository so the layout matches the table below. The scripts expect these paths.

| Path | Contents |
| --- | --- |
| `data/tier1/` | Two short recordings (`mdt_case1.wav`, `mdt_case2.wav`), used by `optimal_transcription/`. |
| `data/tier2/audios/` | Ten full-length clean meeting recordings, `MDT_meeting_full_case{1..10}.wav`. |
| `data/tier2/scripts/` | Reference transcripts (`.txt`) matching each tier-2 recording, used to compute Word Error Rate. |
| `data/tier3/MDT_meeting_full_case{1..10}/` | 127 augmented versions of each case (1,270 files). |
| `data/augmentation/` | Augmentation code (`exhaustive_aug_grid_search.py`), Amazon Polly text-to-speech notebook (`tts_polly.ipynb`) and background noise clips (`noises/`). |

**Tier-3 file naming.** Each file name encodes the augmentations applied, for example `rir_0_noise_7_snr_1_overlap_1_overlapChunklen_1_overlapChunknum_1_dropout_0.wav`:

| Token | Augmentation |
| --- | --- |
| `rir_N` | Room impulse response (reverberation) |
| `noise_N`, `snr_N` | Background noise clip and signal-to-noise level index |
| `overlap_N`, `overlapChunklen_N`, `overlapChunknum_N` | Overlapping speech, chunk length and number of chunks |
| `speed_N`, `pitch_N` | Speed and pitch perturbation |
| `mic` | Microphone simulation |
| `dropout_N` | Signal dropout |

The numeric suffix is the index of the setting level used in `exhaustive_aug_grid_search.py`.

**Third-party material.**
- `RAG/NICE_guidelines/` holds NICE guidelines (CG81, CG164, NG12, NG101), which are © NICE and subject to [NICE's terms](https://www.nice.org.uk/terms-and-conditions).
- The clips in `data/augmentation/noises/` are third-party sound effects and keep their original licences.

These materials are not covered by the University of Hull licence.

## Code Information

| Path | Contents |
| --- | --- |
| `transcription/` | `Whisper_transcribe.py` (Whisper large-v3), `whisperx-daim.ipynb`, `parakeet.ipynb`, `AWS_Transcribe.ipynb`. |
| `optimal_transcription/` | `whisper_onecase_vad.py`: Silero VAD silence removal, DNSMOS-based noise estimation and noise-adaptive decoding, scored with `jiwer`. |
| `optuna/` | `optuna_whisper.py` (Optuna search over Whisper parameters per case), `run_whisper_optuna.sh`, and a SLURM job file `whisper_opt_batch.job`. |
| `RAG/chunking.py`, `RAG/chunk_tables.py` | Section-aware chunking of the NICE PDFs, with table extraction via Camelot. `merged_output.json` is the merged chunk file. |
| `RAG/create_vector_store.py` | Builds a FAISS vector store using `nomic-embed-text` embeddings served by Ollama. |
| `RAG/ollama_retreiver.py` | Retrieves the top-5 guideline chunks for a question. |
| `RAG/medgemma_with_context.py`, `RAG/palmyra-med.py`, `RAG/case_separator_llm.py` | Local `llama.cpp` LLM workflows (MedGemma, Palmyra-Med) that use the retrieved context. |

## Requirements

- Python 3.9 or later; a CUDA GPU is recommended for Whisper and the LLMs.
- `ffmpeg` on the system path.
- For the RAG workflow, a running [Ollama](https://ollama.com) server with the `nomic-embed-text` model.

Python packages, imported across the scripts:

```text
torch  torchaudio  openai-whisper  whisper-normalizer  jiwer  librosa  scipy  numpy  pandas  tqdm
silero-vad  speechmos  optuna  optuna-dashboard  plotly
langchain  langchain-community  langchain-ollama  faiss-cpu  pypdf  camelot-py
llama-cpp-python  huggingface_hub  boto3
```

WhisperX and Parakeet (NeMo) dependencies are installed within their notebooks. The AWS notebooks need AWS credentials. Do not commit credentials.

## Usage Instructions

All scripts use paths relative to the repository root, so they can be run from any directory. Before running:

1. Obtain the data from the authors (available through Figshare upon request) and extract it into `data/` (see [Dataset Information](#dataset-information)). Scripts that need it say so in a comment at the top.
2. For the RAG LLM scripts, put the per-case transcript `.txt` files in `RAG/case_separated/`. For `medgemma_with_context.py`, the case descriptions go in `RAG/case_separated/case_desc/`.

Generated files are written inside the repository (and are git-ignored): `outputs/` (transcripts, Optuna results, LLM outputs), `RAG/chunks/`, `RAG/vector_store/` and `RAG/llm_store/` (downloaded GGUF model weights).

```bash
# 1. Baseline Whisper transcription
python transcription/Whisper_transcribe.py

# 2. Optuna parameter search for one case (1-10)
python optuna/optuna_whisper.py --case 1
#    On a SLURM cluster: sbatch optuna/whisper_opt_batch.job

# 3. Optimised Whisper workflow (VAD + noise-adaptive decoding)
python optimal_transcription/whisper_onecase_vad.py

# 4. RAG: chunk guidelines -> build vector store -> retrieve
python RAG/chunking.py
python RAG/chunk_tables.py
python RAG/create_vector_store.py
python RAG/ollama_retreiver.py

# 5. Local LLM with retrieved guideline context
python RAG/medgemma_with_context.py
```

Notebooks in `transcription/` and `data/augmentation/` can be run in Jupyter or Google Colab.

## Methodology

1. **Data creation.** Scripted MDT meeting dialogues were converted to speech (Amazon Polly) to produce clean recordings (tier 2) with reference transcripts, tier 3 audios were augmented tier 2 audios. For tier 1 recordings, members of the Hull University Teaching Hospitals (HUTH) NHS breast MDT, including surgeons, oncologists, radiologists, pathologists, specialist nurses, and MDT coordinators, simulated a discussion of cases while performing their usual MDT roles. All recordings were fully anonymised, preserving authentic clinical terminology, conversational dynamics, and decision-making without providing identifiable patient information. 
2. **Augmentation.** A grid search over reverberation, background noise at several SNRs, overlapping speech, speed and pitch change, microphone effects and dropout yields 127 variants per case (tier 3).
3. **ASR benchmarking.** Whisper, WhisperX, Parakeet and AWS Transcribe were evaluated by Word Error Rate (`jiwer`, after Whisper English text normalisation).
4. **Whisper optimisation.** Optuna searches the decoding parameters; the optimised workflow adds Silero VAD silence removal and noise-adaptive decoding using DNSMOS.
5. **Guideline RAG.** NICE guidelines are split by section, with tables handled separately, embedded and indexed in FAISS. Top-k chunks are retrieved and supplied to on-device LLMs along with the transcript.

All processing is designed to run locally, with no patient data leaving the device.

## Citations

Third-party resources used: OpenAI Whisper, WhisperX, NVIDIA Parakeet, Silero VAD, DNSMOS, Optuna, LangChain, FAISS, Ollama, llama.cpp, MedGemma, Palmyra-Med, and the NICE guidelines CG81, CG164, NG12 and NG101.

## Funding and Acknowledgements

This study was supported by a grant from the Humber and North Yorkshire Cancer Alliance, Cancer Research and Innovation Award Funding 25-26.

Intellectual property arising from this work is owned by the University of Hull, in line with its IP policy and any funder terms. The research was carried out at the University of Hull.

## License & Contribution Guidelines

**Licence.** Copyright © 2025 University of Hull. All rights reserved. This repository is proprietary software and data, not open source. No licence is granted to use, copy, modify, distribute or create derivative works without prior written permission from the University of Hull. See [LICENSE](LICENSE). The licence does not cover the third-party materials listed under [Dataset Information](#dataset-information).

**Permissions and enquiries.** Contact the corresponding author (a.dhiman@hull.ac.uk) or the University of Hull research and innovation office.

**Contributions.** Contributions are not accepted unless agreed in writing in advance. Contributors would be required to assign or license their contributions to the University of Hull.
