#!/usr/bin/python3

import os
import numpy as np
import json
try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import glob
import torch
import whisper
import pandas as pd
import torchaudio

from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



model = whisper.load_model("large-v3")
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)



class MyAUGDataset(torch.utils.data.Dataset):
    """
    Custom dataset for local audio + transcripts in LibriSpeech-style preprocessing.
    """
    def __init__(self, audio, scripts,  file_name, device="cpu",):
        self.audio = audio
        self.scripts = scripts
        self.device = device
        self.file_name = file_name
        # collect audio files
        self.audio_files = sorted([f for f in os.listdir(audio) if f.endswith(".wav")])
        
        #print (self.audio_files)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        
        #base = os.path.splitext(audio_file)[0]
        
        #text_file = 'MDT_meeting_full_case2.txt'
        text_file = self.file_name
        
        # Load audio
        audio, sample_rate = torchaudio.load(os.path.join(self.audio, audio_file))
        assert sample_rate == 16000, "Whisper requires 16kHz audio"

        # Flatten to mono, pad/trim to 30s, move to device
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)

        # Convert to log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio, n_mels=128)

        # Load transcription
        with open(os.path.join(self.scripts, text_file), "r") as f:
            text = f.read().strip()
        return mel, text, audio_file
    



scripts = glob.glob('simulated_script/MDT_meeting_full_case*')

for script in scripts[:1]:
    print ('============================')
    print (script)
    path = script.split('/')[-1].replace('.txt', '')
    
    dataset = MyAUGDataset(audio=f"augmented_audio/{path}",  scripts= "simulated_script", file_name=path+str('.txt'),device=DEVICE)

    print("Total samples:", len(dataset))

    mel, text, ids = dataset[0]
    print("Mel shape:", mel.shape)   # e.g., torch.Size([80, 3000])
    print("Transcript:", text)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    # predict without timestamps for short-form transcription
    options = whisper.DecodingOptions(language="en", without_timestamps=False)
    
    hypotheses = []
    references = []
    file_ids = []
    timestamps = []
    


    
    for mels, texts, ids in tqdm(loader):
        results = model.decode(mels, options)
        hypotheses.extend([result.text for result in results])
        references.extend(texts)
        file_ids.extend(ids)
        
        print (results)
        if hasattr(results, "segments") and results.segments is not None:
            ts = [{"start": seg["start"], "end": seg["end"], "text": seg["text"]}
                  for seg in results.segments]
        else:
            ts = []
            
        timestamps.append(ts)
        
    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references, file_ids= file_ids, timestamps= timestamps))
    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]
    
  
    wil = WordInfoLost()

    # Apply row-wise WER calculation
    data["WER"] = data.apply(
        lambda row: jiwer.wer(row["reference_clean"], row["hypothesis_clean"]),
        axis=1
    )
    data["WIL"] = data.apply(
        lambda row: jiwer.wil(row["reference_clean"], row["hypothesis_clean"]),
        axis=1
    )

    # If you still want the average WER across rows
    print("Average WER: {:.2f}%".format(data["WER"].mean() * 100))

    # If you still want the average WER across rows
    print("Average WER: {:.2f}%".format(data["WIL"].mean() * 100))
    
    
    data.to_csv(f'outputs/Whisper_augmented/with_timestampstranscriptions_{path}.csv')