from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import re
from pathlib import Path

# Paths are relative to the repository root, so the script can be run from anywhere.
from pathlib import Path as _P
REPO_ROOT = _P(__file__).resolve().parents[1]
RAG_DIR = REPO_ROOT / 'RAG'
# Case transcripts (.txt, one per case): place them in RAG/case_separated/ - use the outputs of
# optimal_transcription/ or the data from Figshare (https://figshare.com/s/27501bb3e1096f720c55).
CASE_DIR = RAG_DIR / 'case_separated'
LLM_DIR = RAG_DIR / 'llm_store'  # GGUF model weights are downloaded here from Hugging Face
OUT_DIR = REPO_ROOT / 'outputs' / 'palmyra-med'
OUT_DIR.mkdir(parents=True, exist_ok=True)

#####reading data
folder = CASE_DIR
texts = {}

for txt_file in folder.glob("*.txt"):
    with txt_file.open("r", encoding="utf-8") as f:
        texts[txt_file.stem] = f.read()


'''txt_file = "/home/data/LLM/real_case/whisper_combination/whisper_case1_silremoved_anonym_tuned_noise_th3.5.txt"

with open(txt_file, 'r') as f:
	texts = f.read()
'''


#print (texts)



#print (texts)
                             
    
        
#model_name = "aaditya/OpenBioLLM-Llama3-8B-GGUF"  #approx 5 Gb
#model_file= "openbiollm-llama3-8b.Q5_K_M.gguf"



model_name = "bartowski/Palmyra-Med-70B-32K-GGUF"  
model_file= "Palmyra-Med-70B-32K-IQ3_M.gguf"



model_path = hf_hub_download(model_name, filename=model_file, local_dir = str(LLM_DIR), local_dir_use_symlinks=False)
print ("My model path: ", model_path)

llm= Llama(model_path=model_path, n_gpu_layers= -1, n_ctx=4096)



for k in list(texts.keys()):

	print(f"reading file:{k}")
	#print (texts[k])
	print ('-------------------------')
	
	
	#for k in [texts]:
	#Question = "So, 44-year-old lady, symptomatic, presented with a grade 2 IDC, ER positive, ER positive, HER2 negative. We're discussing post-op results. She underwent a bright breast, quadrocolexation, central elenobiasis, and a perforated flap. She has 38mm grade 2 invasive breast carcinoma with intermediate grade DCIS, ERPR positive, HER2 negative. Three nodes, all negative, closed anterior margin, otherwise excised. So, we're happy with the anterior margin. Surgery is completed. So, she needs long-cap Dx."
	
	Question = texts[k]

	sys_prompt= f"""

	You are a clinical MDT transcription assistant specialized in breast oncology. 

	  

	Your tasks: 

	1. Consider the given text as the MDT meeting discussion transcript of a single case.
	2. Your job is to return the FULL original transcript, word-for-word, in the SAME ORDER. You may ONLY change a word or phrase if it is clearly an error.  
	
	3. For the given case, you may clean and normalise the language by correcting spelling, grammar and punctuations without including/excluding any information. with your breast cancer expertise, you have to clean the medical terminologies wherever possible, by correcting ASR phonetic drift into medically valid terms. 

	4. For the case, you have to extract the case description, you MUST NOT include or modify any information not found in the given text. You HAVE TO only use the information provided in the text with corrections.  

	5. The case description may include patient’s features/conditions like, age, gender, demographics, diagnosis, pathology, imaging, nodal status, biomarkers etc.  

	6. For each case, you also have to extract the Treatment Decisions made during the MDT meeting, using this text. you MUST NOT include or modify any information not found in the given text. You HAVE TO only use the information provided in the text. 

	7. The treatment decisions may include agreed MDT plan, investigations, referrals, follow-up actions etc.  

	8. For both case description and treatment plan, you have to use the EXACT VERBATIM from the provided text, without any structural modifications. 

	9. IF the details are not clear, mark it as “UNCLEAR in transcript”.   

	Output format (STRICT): 

	For each case, use the following structure: 

	  

	CASE X: 
	
	Corrections made:

	Case Description:  

	  

	Treatment Decisions: 

	 

	=============================== 

	 EXAMPLE 

	INPUT TRANSCRIPT: 
	" So, 44-year-old lady, symptomatic, presented with a grade 2 IDC, ER positive, ER positive, HER2 negative. We're discussing post-op results. She underwent a bright breast quadrantectomy, central lymph node biopsy, and a perforator flap. She has 38-millimeter grade 2 invasive breast carcinoma with intermediate grade DCIS, ERPR positive, HER2 negative, 3 nodes all negative, close anterior margin otherwise excised. So we're happy with anterior margin, surge is completed. So she’s long-capped yet." 
	 
	OUTPUT: 
	
	Corrections made: removed repetition of ER positive,  'bright' changed to 'right', 'quadrantectomy' changes to 'wide local excision', 'central lymph node' changes to 'sentinel node', 'perforator flap' changes to 'perforated flap', '3' changes to 'three', 'close' changes to 'closed', 'surge' changes to 'surgery', 'long-capped yet' changes to 'needs an oncotype Dx'.
	
	Case Description: 
	44-year-old lady, symptomatic, presented with a grade 2 IDC, ER positive, HER2 negative. We're discussing post-op results, she underwent right breast, wide local excision, sentinel lymph node biopsy and a perforated flap. She has 38 mm grade 2 invasive breast carcinoma with intermediate grade DCIS, ERPR positive, HER2 negative, three nodes all negative, closed anterior margin, otherwise excised. So we're happy with anterior margin, surgery is completed. 

	 
	Treatment Decisions: 
	she needs an oncotype Dx. 
	--- 

	Now process the following MDT transcript: 

	""" 
	
	user_prompt = Question
	#response = llm(prompt, max_tokens= 1024)['choices'][0]['text']

	response = llm.create_chat_completion(messages =[{"role":"system","content": sys_prompt}, {"role":"user","content": user_prompt}],
	temperature= 0.02, top_p =0.6,  min_p =0.15, repeat_penalty= 1.15, frequency_penalty =0.1, typical_p = 0.7,  top_k = 15, mirostat_mode = 0, tfs_z = 1.0, max_tokens= 4000)['choices'][0]['message']['content']
	
	
	print ('\n\n\n', response)

	with open(OUT_DIR / f'{k}.txt' , 'w') as f:
		f.write(response)

