import whisper
import os

# Paths are relative to the repository root, so the script can be run from anywhere.
# DATA: download the dataset from Figshare (https://figshare.com/s/27501bb3e1096f720c55)
# and extract it into the repository's data/ folder (data/tier1, data/tier2, data/tier3).
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / 'data' / 'tier3'  # augmented audio, one folder per case (Figshare)
SCRIPT_DIR = REPO_ROOT / 'data' / 'tier2' / 'scripts'  # reference transcripts (Figshare)
OUT_DIR = REPO_ROOT / 'outputs' / 'optuna_whisper'
OUT_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np
import json
import glob
import jiwer
import pandas as pd
from tqdm import tqdm
import torch
import optuna
from whisper_normalizer.english import EnglishTextNormalizer
import argparse
from threading import Lock
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline
from optuna_dashboard import save_plotly_graph_object
import os


##pkill -f optuna-dashboard

os.environ['PYTHONIOENCODING'] ='utf-8'

parser =argparse.ArgumentParser()
parser.add_argument('--case', type=int, default=1)
args = parser.parse_args()

print ("current case:", args.case)


device= "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)



model = whisper.load_model("large-v3", device=device)
normalizer = EnglishTextNormalizer()

i = int(args.case)

model_lock = Lock()

# --------------------------------------------------
# Utility: run Whisper with tunable parameters
# --------------------------------------------------
def run_transcription(audio_file, decoding_params):
    """
    Runs Whisper with Optuna-suggested decoding parameters.
    """
    with model_lock:
        result = model.transcribe(
            audio_file,
            initial_prompt=decoding_params["initial_prompt"],
            fp16=False,
            temperature=decoding_params["temperature"],
            beam_size=decoding_params["beam_size"],
            best_of=decoding_params["best_of"],
            compression_ratio_threshold = decoding_params["compression_ratio_threshold"],
            logprob_threshold = decoding_params["logprob_threshold"],
            length_penalty=decoding_params["length_penalty"], 
            carry_initial_prompt = decoding_params["carry_initial_prompt"], 
            hallucination_silence_threshold = decoding_params["hallucination_silence_threshold"]
        )
    return result["text"]



    



# --------------------------------------------------
# Optuna Objective Function
# --------------------------------------------------
def objective(trial):

    # Define search space
    decoding_params = {
        "temperature": trial.suggest_float("temperature", 0.0, 1.0),
        "beam_size": trial.suggest_int("beam_size", 1, 5),
        "best_of": trial.suggest_int("best_of", 1, 5),
        "compression_ratio_threshold": trial.suggest_float("compression_ratio_threshold", 1.5, 3.5),
        "logprob_threshold": trial.suggest_float("logprob_threshold", -2.0, -0.5),
        "length_penalty": trial.suggest_float("length_penalty", 0.0, 1.0),
        "initial_prompt": trial.suggest_categorical("initial_prompt", ["", "This audio is a clinical MDT (Multidisciplinary Team) meeting involving multiple healthcare professionals such as consultants, surgeons, oncologists, radiologists, pathologists, CNS nurses, and coordinators. The discussion includes patient cases, imaging findings, pathology reports, diagnosis, staging, treatment planning, surgical considerations, radiotherapy, chemotherapy, follow-up plans, and risk factors. Use correct UK medical terminology. Preserve important clinical details such as measurements, staging (e.g., T2N1M0), biomarkers (ER, PR, HER2), lab values, tumour size, margins, lymph nodes, and anatomical locations. Do NOT simplify medical language. Transcribe medical terms accurately, including complex drug names (e.g., Pembrolizumab, Letrozole), procedures (e.g., mastectomy, resection), and abbreviations (MDT, MRI, CT, PET, MDT coordinator, CNS). Do not delete or make up any information."]),
        "carry_initial_prompt": trial.suggest_categorical("carry_initial_prompt", [True, False]), 
        "hallucination_silence_threshold": trial.suggest_float("hallucination_silence_threshold", 0.0, 0.3)
    }
    
        
    # Pick a small number of samples per trial for speed
    scripts = glob.glob(str(AUDIO_DIR / 'MDT_meeting_full_case*'))  # <-- EDIT: use more for thorough tuning
    print (len(scripts))
    all_wers = []

    for idx, script in enumerate(scripts):
        case = os.path.basename(script)
        audio_path = glob.glob(script+'/*')
        print (case)
        
        with open(SCRIPT_DIR / f'{case}.txt') as f:
            ref_text = f.read()
        
        ref_clean = normalizer(ref_text)
        
        print ('all files')
        print (len(audio_path))

        for audio_file in tqdm(audio_path):# <-- EDIT: small sample per trial

            print ('====================AUDIO===================')
            #print (case+'\n')
            print (audio_file)
            
            
            hyp = run_transcription(audio_file, decoding_params)

            hyp_clean = normalizer(hyp)

            wer = jiwer.wer(ref_clean, hyp_clean)
            print ('WER:', wer)
            all_wers.append(wer)
            
            intermediate_value = float(np.mean(all_wers))
            trial.report(intermediate_value, step=idx)
            
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Optuna minimizes — so return mean WER
    return float(np.mean(all_wers))



# --------------------------------------------------
# Run Optuna Study
# --------------------------------------------------

pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=0, interval_steps=1)
study = optuna.create_study(study_name="whisper_optuna_all", direction="minimize", storage="sqlite:///whisper_optuna_all.db", load_if_exists=True )

num_threads = os.cpu_count()

print ('\n Number of threads:\t', num_threads)

study.optimize(objective, n_trials=50, n_jobs = num_threads)   # <-- EDIT number of trials

print("\n=================================== Best Parameters ===")
if len(study.trials)>0:
    print(study.best_params)
    print("Best WER:", study.best_value)
else:
    print ("no completed trials yet. run more!")


with open(OUT_DIR / 'best_params_all.txt', 'w') as f:
    f.write(f'working for case ALL \n')
    f.write(json.dumps(study.best_params, indent=2))
    f.write('\n=======\n Best WER: \t ')
    f.write(str(study.best_value))
    
figure1 = optuna.visualization.plot_contour(study)
save_plotly_graph_object(study, figure1)

figure2 = optuna.visualization.plot_edf(study)
save_plotly_graph_object(study, figure2)

figure3 = optuna.visualization.plot_intermediate_values(study)
save_plotly_graph_object(study, figure3)

figure4= optuna.visualization.plot_optimization_history(study)
save_plotly_graph_object(study, figure4)

figure5= optuna.visualization.plot_parallel_coordinate(study)
save_plotly_graph_object(study, figure5)

figure6= optuna.visualization.plot_param_importances(study)
save_plotly_graph_object(study, figure6)

figure7= optuna.visualization.plot_rank(study)
save_plotly_graph_object(study, figure7)

figure8= optuna.visualization.plot_slice(study)
save_plotly_graph_object(study, figure8)

figure9= optuna.visualization.plot_timeline(study)
save_plotly_graph_object(study, figure9)


# --------------------------------------------------
# FINAL EVALUATION ON FULL DATASET USING BEST PARAMS
# --------------------------------------------------

'''
print ('\n\n--------------Running best param on full dataset------------')

best_params = study.best_params

scripts = glob.glob(str(AUDIO_DIR / f'MDT_meeting_full_case{i}*'))
all_results = []

for script in tqdm(scripts):
    print("\n============================")
    print(script)

    case = os.path.basename(script)
    audio_files = glob.glob(script + '/*')

    with open(SCRIPT_DIR / f'{case}.txt') as f:
        ref_text = f.read()

    references = []
    hypotheses = []
    file_ids = []

    for audio_file in tqdm(audio_files):
        audio_id = audio_file.split("/")[-1].replace('.wav', '')

        hyp = run_transcription(audio_id, best_params)

        references.append(ref_text)
        hypotheses.append(hyp)
        file_ids.append(audio_id)

    df = pd.DataFrame({
        "file_id": file_ids,
        "hypothesis": hypotheses,
        "reference": references
    })

    df["hypothesis_clean"] = df["hypothesis"].apply(normalizer)
    df["reference_clean"] = df["reference"].apply(normalizer)

    df["WER"] = df.apply(lambda r: jiwer.wer(r["reference_clean"], r["hypothesis_clean"]), axis=1)
    df["WIL"] = df.apply(lambda r: jiwer.wil(r["reference_clean"], r["hypothesis_clean"]), axis=1)

    print(f"Case {case}: Mean WER = {df['WER'].mean():.3f} (+/- {df['WER'].std():.3f})")

    out_path = str(OUT_DIR / 'viper' / f'{case}.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    all_results.append(df)



# --------------------------------------------------
# Save Optuna Study
# --------------------------------------------------
study_df = study.trials_dataframe()
study_df.to_csv(str(OUT_DIR / 'results' / f'{case}.csv'), index=False)

print("\nSaved study results as optuna_results.csv")
'''