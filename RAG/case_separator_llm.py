from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import re
from pathlib import Path

# Paths are relative to the repository root, so the script can be run from anywhere.
from pathlib import Path as _P
REPO_ROOT = _P(__file__).resolve().parents[1]
RAG_DIR = REPO_ROOT / 'RAG'
# Case transcripts (.txt, one per case): place them in RAG/case_separated/ - use the outputs of
# optimal_transcription/ or the data available through Figshare on request.
CASE_DIR = RAG_DIR / 'case_separated'
LLM_DIR = RAG_DIR / 'llm_store'  # GGUF model weights are downloaded here from Hugging Face
OUT_DIR = REPO_ROOT / 'outputs' / 'medgemma' / 'case_separator'
OUT_DIR.mkdir(parents=True, exist_ok=True)


#####reading data
folder = CASE_DIR


texts = {}

for txt_file in folder.glob("*.txt"):
    with txt_file.open("r", encoding="utf-8") as f:
        texts[txt_file.stem] = f.read()






#print (texts)
                             
    
        
#model_name = "aaditya/OpenBioLLM-Llama3-8B-GGUF"  #approx 5 Gb
#model_file= "openbiollm-llama3-8b.Q5_K_M.gguf"



model_name = "unsloth/medgemma-27b-text-it-GGUF"  
model_file= "medgemma-27b-text-it-Q4_K_S.gguf"



model_path = hf_hub_download(model_name, filename=model_file, local_dir = str(LLM_DIR), local_dir_use_symlinks=False)
print ("My model path: ", model_path)

llm= Llama(model_path=model_path, n_gpu_layers= -1, n_ctx=4096)


for k in list(texts.keys()):

	print(f"reading file:{k}")
	print (texts[k])
	print ('-------------------------')

	#Question = "So, 44-year-old lady, symptomatic, presented with a grade 2 IDC, ER positive, ER positive, HER2 negative. We're discussing post-op results. She underwent a bright breast, quadrocolexation, central elenobiasis, and a perforated flap. She has 38mm grade 2 invasive breast carcinoma with intermediate grade DCIS, ERPR positive, HER2 negative. Three nodes, all negative, closed anterior margin, otherwise excised. So, we're happy with the anterior margin. Surgery is completed. So, she needs long-cap Dx."
	
	Question = texts[k]

	prompt= f"""
	<start_of_turn>user 
	You are a clinical text normalization assistant. You will be shown ASR-generated medical MDT text and the corresponding ground truth. Your task is to learn how to correct errors by mapping incorrect phrases to their normalized clinical form. 
	Below are examples of corrections:
	EXAMPLE 1

	ASR → GROUND TRUTH CORRECTIONS:
	- "bright breast" → "right breast"
	- "quadrocolexation" → "wide local excision"
	- "central elenobiasis" → "sentinel lymph node biopsy"
	- "long-cap Dx" → "Oncotype Dx"
	- "fibroid enoma" → "fibroadenoma"
	- "dachshund region" → "axillary region"
	- "non-fibroid enoma" → "known fibroadenoma"
	- "hair too low" → "HER2 low"
	- "heaping of the breast and the axilla" → "clipping of the breast and the axilla"
	- "predict" → "PREDICT"
	

	RULES LEARNED:
	- Correct spelling, grammar, and punctuation 
	- Always restore formal clinical terminology 
	- Prefer standard oncology abbreviations (ER, PR, HER2, NACT, DCIS, LCIS) 
	- Standardize clinical terminology using formal medical language 
	- Preserve all clinical meaning, decisions, and intent 
	- Maintain a professional, structured clinical note style 
	- Do NOT add new clinical information 
	- Do NOT remove important medical details 
	- Keep patient-identifiable information unchanged if present 
	- Correct ASR phonetic drift into medically valid terms 
	- Preserve numeric and pathological values exactly
	
	Output format: 
	- DO NOT change the structure of the text at all.  
	- The order of words should not change. 

	Now apply the same correction style to the next input.

	\"\"\" 

  

	{Question} 

	\"\"\" 
	
	<end_of_turn>
	<start_of_turn>model
	"""

	response = llm(prompt, max_tokens= 1024)['choices'][0]['text']

	print ('\n\n\n', response)

	with open(OUT_DIR / f'{k}.txt' , 'w') as f:
		f.write(response)

