import whisper
import os
import numpy as np
import json
import subprocess
try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass
import scipy
import glob
import jiwer
import pandas as pd
from tqdm import tqdm
import torch
from whisper_normalizer.english import EnglishTextNormalizer
from scipy.io.wavfile import read, write
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio
import torchaudio
from speechmos import dnsmos
import librosa
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)




model = whisper.load_model("large-v3", device=device)


#for DNSMOS

os.environ['KMP_AFFINITY'] = 'disabled'
os.environ['OMP_PROC_BIND'] = 'false'
os.environ['OMP_DISPLAY_ENV'] = 'true'

num_cores = os.getenv('SLURM_CPUS_PER_TASK', '4')

os.environ['OMP_NUM_THREADS'] = str(num_cores)
os.environ['MKL_NUM_THREADS'] = str(num_cores)
##activate whisper-conda env to run this and load ffmpeg

torch.set_num_threads(int(num_cores))
#model = whisper.load_model("large-v3", device=device)

audio_file = 'real_case/mdt_case2.wav'





####Using VAD

#torchaudio.set_audio_backend("sox_io")
vad_model = load_silero_vad()
'''(get_speech_timestamps, _, read_audio, _, _) = utils'''

audio, sr = torchaudio.load(audio_file)

print ('=============================================================================')



#audio = audio.astype(np.float32)

if audio.shape[0]>1:
    audio = audio.mean(dim=0)
else:
    audio = audio[0]
    
    
audio_np =audio.numpy()

if sr!=16000:
    audio = scipy.signal.resample_poly(audio_np, 16000, sr)
    sr =16000
    
audio_np = read_audio(audio_file)


'''mos_scores = dnsmos.run(audio_file, sr=16000)
results ={}

results['DNSMOS_overall'] = mos_scores['ovrl_mos']
results['DNSMOS_speech_quality'] = mos_scores['sig_mos']
results['DNSMOS_noise_quality'] = mos_scores['bak_mos']'''





print ('=============================================================================')



speech_timestamps = get_speech_timestamps(audio_np, vad_model, sampling_rate =sr)



#print ('speech timestamp')
#print (speech_timestamps)

##compute speech ratio

speech_duration= sum((ts['end']-ts['start']) for ts in speech_timestamps)/ sr
#print ('speechduration')
#print (speech_duration)
total_duration = len(audio)/sr
speech_ratio = speech_duration/max(total_duration, 1e-6)


##estimate noise


def estimate_snr(audio, speech_timestamps, sr):
    audio = np.asarray(audio, dtype = np.float32)
    mask = np.ones(len(audio), dtype=bool)
    
    for ts in speech_timestamps:
        mask[ts['start']:ts['end']] =False
    
    noise = audio[mask]
    
    if len(noise)<sr*0.1:
        return 30.0
    
    noise_rms = np.sqrt(np.mean(noise**2))
    signal_rms = np.sqrt(np.mean(audio**2))
    snr = 20*np.log10(signal_rms/(noise_rms+1e-8))
    return snr

#hard threshold approach may not work for case 1 with snr = 15.05
def is_noisy_audio(snr, speech_ratio, snr_threshold=15, speech_ratio_threshold=0.6): #snr >20 means clean, studio like sppech 
    #speech ratio >=0.6 also clean

    return (snr<snr_threshold) or (speech_ratio <speech_ratio_threshold)


#using soft score 

def noise_score(snr, speech_ratio):
    #snr_score = np.clip((20-snr)/10, 0, 1) #using normalisation =>keeping the range of 10-20
    
    snr_score= 1/(1+np.exp((snr-18)/2))
    print ("snr_score")
    print (snr_score)
    #sr_score= np.clip((0.5- speech_ratio)/0.3, 0,1)
    sr_score= 1/(1+np.exp((speech_ratio-0.6)/0.15))
    print ("sr_score")
    print (sr_score)
    return 0.6*snr_score +0.4*sr_score #snr correlated more strongly with the ASR errors, which leads to haullcinations and repetitions, speech ratio is for silences, background music etc


snr = estimate_snr(audio, speech_timestamps, sr)
score = noise_score(snr, speech_ratio)

print ("snr")
print (snr)
print ("speech ratio")
print (speech_ratio)
print ('soft score')
print (score)
if score>0.5:
    mode="noisy"
else:
    mode ="clean"


is_noisy = is_noisy_audio(snr, speech_ratio)


print (snr)

print ("hard threshold")
print (is_noisy)
print ("soft")
print (mode)


# function to use ffmpeg in python and get the detected silence.
def detect_silence(path,time):
    '''
    This function is a python wrapper to run the ffmpeg command in python and extranct the desired output
    
    path= Audio file path
    time = silence time threshold
    
    returns = list of tuples with start and end point of silences
    
    '''
    #command="ffmpeg -i "+path+" -af silencedetect=n=-23dB:d="+str(time)+" -f null -"
    
    command= [
        "ffmpeg",
        "-i", path, 
        "-af", f"silencedetect=n=-23dB:d={time}",
        "-f", "null",
        "-"
    ]
    
    
    
    out = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    s=stdout.decode("utf-8")
    k=s.split('[silencedetect @')
    if len(k)==1:
        #print(stderr)
        return None
        
    start,end=[],[]
    for i in range(1,len(k)):
        x=k[i].split(']')[1]
        if i%2==0:
            x=x.split('|')[0]
            x=x.split(':')[1].strip()
            end.append(float(x))
        else:
            x=x.split(':')[1]
            x=x.split('size')[0]
            x=x.replace('\r','')
            x=x.replace('\n','').strip()
            start.append(float(x))
    return list(zip(start,end))


def remove_silence(file,sil,keep_sil,out_path):
    '''
    This function removes silence from the audio.
    
    Input: 
    file = Input audio file path
    sil = List of silence time slots that needs to be removed
    keep_sil = Time to keep as allowed silence after removing silence
    out_path = Output path of audio file
    
    returns:
    Non - silent patches and save the new audio in out path
    '''
    rate,aud=read(file)
    a=float(keep_sil)/2
    sil_updated=[(i[0]+a,i[1]-a) for i in sil]
    
    # convert the silence patch to non-sil patches
    non_sil=[]
    tmp=0
    ed=len(aud)/rate
    for i in range(len(sil_updated)):
        non_sil.append((tmp,sil_updated[i][0]))
        tmp=sil_updated[i][1]
    if sil_updated[-1][1]+a/2<ed:
        non_sil.append((sil_updated[-1][1],ed))
    if non_sil[0][0]==non_sil[0][1]:
        del non_sil[0]
    
    # cut the audio
    print('slicing started')
    ans=[]
    ad=list(aud)
    for i in tqdm(non_sil):
        ans=ans+ad[int(i[0]*rate):int(i[1]*rate)]
    #nm=path.split('/')[-1]
    write(out_path,rate,np.array(ans))
    return non_sil, out_path

audio_silences = detect_silence(audio_file, 5)

if audio_silences is None or len(audio_silences) ==0:
    print ('no silence detected')
    trimmed_audio_path = audio_file
else:
    trim_audio, trimmed_audio_path= remove_silence(audio_file, audio_silences, 2, 'real_case/case1_anonym_silence_removed2.wav')             

temperature=0.7596350092541306
beam_size=2
best_of=2
compression_ratio_threshold = 1.5323007177495684
logprob_threshold = -1.0903724240294121
length_penalty=0.7594738864766823
hallucination_silence_threshold = 0.22902235239020646

#audio_id = audio_file.replace('.wav','')


#audio_file = 'real_case/no_silence.wav'

mos_scores = dnsmos.run(audio_file, sr=16000)
        
noise_level = mos_scores['bak_mos']

for th in [-1]: #keep th =6 for using tuned without noise threshold, keep th=-1 for no tuning.
    if noise_level >=th:
        print ('running not tuned')
        result = model.transcribe(audio_file,fp16=False, language='en' ) #change trimmed audio path to audio_file for no silence removal 
    else:
        print ('running tuned')
        result = model.transcribe(audio_file,fp16=False, language='en', temperature = temperature,beam_size= beam_size,best_of= best_of,compression_ratio_threshold= compression_ratio_threshold, logprob_threshold= logprob_threshold, length_penalty= length_penalty, hallucination_silence_threshold= hallucination_silence_threshold ) #change trimmed audio path to audio_file for no silence removal 

    #result = model.transcribe(trimmed_audio_path,fp16=False, language='en', temperature = temperature, beam_size= beam_size, best_of= best_of, compression_ratio_threshold= compression_ratio_threshold, logprob_threshold= logprob_threshold, length_penalty= length_penalty, hallucination_silence_threshold= hallucination_silence_threshold)

    #result = model.transcribe(audio_file,fp16=False, language='en')

    #result = model.transcribe(trim_audio,fp16=False, language='en', temperature=0, beam_size=3, best_of=3, compression_ratio_threshold=2.0, logprob_threshold =-1.0, condition_on_previous_text = False)

    #print (result["text"])

    with open(f'real_case/whisper_case2{th}.txt', 'w') as f:
        f.write(result['text'])

