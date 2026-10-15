##run on llama_cpp container

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import re
from pathlib import Path
from langchain_core.documents import Document
import os
import json
from langchain_community.vectorstores.faiss import FAISS

from langchain_ollama import OllamaEmbeddings

####loading from stored retreived documents 
'''def load_retrieved(filepath):
    results = []
    with open(filepath, "r") as f:
        for line in f:
            item = json.loads(line)
            docs = [
                Document(
                    page_content=d["content"],
                    metadata=d["metadata"]
                )
                for d in item["docs"]
            ]
            ret_question =item["question"] 
            results.append(docs)
    return ret_question, results
    
    
retrieved_ques, retrieved_docs = load_retrieved('/home/data/LLM/LLM_outputs/medgemma/real_case/medgemma_8kxl/ollama_retrieved/test.json')

print (retrieved_docs[0][0].page_content)'''




#####reading transcription data
folder = Path("/home/data/LLM/case_separated/case_desc")
#folder = Path("/home/data/LLM/real_case_groundtruth/treatment_prediction/case_desc_gt/")


texts = {}

for txt_file in folder.glob("*.txt"):
    with txt_file.open("r", encoding="utf-8") as f:
        texts[txt_file.stem] = f.read()


#txt_file = "/home/data/LLM/real_case/whisper_combination/whisper_case1_silremoved_anonym_tuned_noise_th3.5.txt"

'''with open(txt_file, 'r') as f:
	texts = f.read()'''

##retreiving dynamically. 

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = FAISS.load_local(
    '/home/data/LLM/vector_store/faiss_nomic_ollama',
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})






#print (texts)
                             
    
        
#model_name = "aaditya/OpenBioLLM-Llama3-8B-GGUF"  #approx 5 Gb
#model_file= "openbiollm-llama3-8b.Q5_K_M.gguf"



model_name = "unsloth/medgemma-27b-text-it-GGUF"  
model_file= "medgemma-27b-text-it-UD-Q8_K_XL.gguf"

local_path = '/home/data/LLM/llm_store/medgemma-27b-text-it-UD-Q8_K_XL.gguf'


if not os.path.exists(local_path):
	print ('downloading model-----')
	local_path = hf_hub_download(
        repo_id=model_name,
        filename=model_file,
        local_dir='/home/data/LLM/llm_store/',
        local_dir_use_symlinks=False)



#model_path = hf_hub_download(model_name, filename=model_file, local_dir = '/home/data/LLM/llm_store/', local_dir_use_symlinks=False)
print ("My model path: ", local_path)

llm= Llama(model_path=local_path, n_gpu_layers= -1, n_ctx=8192)



def build_prompt(question, retrieved_docs):
	#context = "\n\n".join([doc.page_content for doc in retrieved_docs[0]])  ##use when loading from stored retrieved content
	context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    	
	
	#Question = k

	sys_prompt= f"""

	SYSTEM ROLE:
	You are a clinical decision-support assistant for multidisciplinary team (MDT) meetings.
	You do NOT provide medical advice.
	You summarize cases and map them to high-level NICE guideline pathways.

	TASK:
	From the transcript below:
	1. Understand the case description carefully, ONLY consider information that is present. 
	2. Propose a TREATMENT PLAN that is CONSISTENT WITH NICE BREAST CANCER GUIDELINES

	RULES:
	- Use ONLY information explicitly stated in the transcript
	- Do NOT assume staging, performance status, or comorbidities
	- Do NOT invent biomarkers, imaging results, or pathology
	- If required information is missing, write: INSUFFICIENT INFORMATION
	- Treatment plans must be guideline-aligned, not personalized prescriptions
	- Use conditional language: "may be considered", "is typically recommended", "depending on eligibility"

	OUTPUT FORMAT: 
	
	  "Case Description":
	
	  "Proposed Treatment Plan": 
	  
	  "guideline basis": 
	  
	  "confidence": "low | medium | high"
	

	IMPORTANT:
	- If multiple patients are present, output an ARRAY (one per case)
	- If no clear treatment decision can be derived, populate treatment_plan with "INSUFFICIENT INFORMATION"
	
	
	==============================================
	
	EXAMPLE:
	
	Case Description: This lady is 46, she's premenopausal, she's had a left wide local excision and sentinel node biopsy. 30 mm, invasive lobular, grade 2, with pleomorphic LCIS, ER6, PR8, and HER2 negative on FISH, 3 nodes, all negative, and it's completely excised.
	
	"Proposed Treatment Plan":
	
	So she needs an Oncotype Dx? Oncotype Dx. For Oncotype DX, that was your question. So, remove the plan; we will rediscuss in 3 weeks. Thank you. Hold on, do we want to discuss the following treatment as well, at least, after Oncotype DX? In terms of the Oncotype DX? Chemo versus no chemo would be decided on, but anything that she would, this patient would get an adjuvant? Well, she would get radiotherapy to the breast and tamoxifen, but as I say, if Oncotype DX is high, she might get ovarian suppression. and an AI, and Ribociclib, so it makes a big difference.
	
	 "guideline basis": "Treatment planning is consistent with NICE NG101 and DG34. Oncotype DX is used to guide adjuvant chemotherapy decision-making in ER-positive, HER2-negative, node-negative early breast cancer where recurrence risk is uncertain. Adjuvant whole-breast radiotherapy is recommended following breast-conserving surgery. Endocrine therapy with tamoxifen is standard in premenopausal patients, with consideration of ovarian suppression and aromatase inhibitor in higher-risk disease. If high-risk features are confirmed, adjuvant ribociclib may be considered in line with NICE TA872."
	
	"confidence": "high"
	
	
	""" 

	user_prompt = (
        	f"Context:\n{context}\n\n"
        	f"Question:\n{question}"
        	)
	return sys_prompt, user_prompt
	
def save_retrieved(question, docs,k):
	data = {
        "question": question,
        "docs": [
            {
                "content": d.page_content,
                "metadata": d.metadata
            }
            for d in docs
        ]
        }

	with open(f'/home/data/LLM/LLM_outputs/medgemma/real_case/medgemma_8kxl/ollama_retrieved/transcr{k}.json' , "w") as f:
        	f.write(json.dumps(data) + "\n")
        	
	
def ask_rag(question, retrieved_docs):

	'''if question != retrieved_ques: ##use when loading from stored retrieved content
		print ("ERROR: The questions don't match")
		return'''  
		
	retrieved_docs = retrieved_docs
	
	# Build prompt
	sys_prompt, user_prompt = build_prompt(question, retrieved_docs)
	
	print (sys_prompt)
	response = llm.create_chat_completion(messages =[{"role":"system","content": sys_prompt}, {"role":"user","content": user_prompt}],temperature= 0.1, top_p =0.9,  min_p =0.05, repeat_penalty= 1.1, frequency_penalty =0.0, typical_p = 1.0,  top_k = 40, mirostat_mode = 0, tfs_z = 1.0, max_tokens= 3000)['choices'][0]['message']['content']

#params for few shot- temperature= 0.05, top_p =0.7,  min_p =0.05, repeat_penalty= 1.2, frequency_penalty =0.3, typical_p = 0.7,  top_k = 20, mirostat_mode = 0, tfs_z = 1.0, 
#params for zero shot - temperature= 0.1, top_p =0.8,  min_p =0.05, repeat_penalty= 1.25, frequency_penalty =0.4, typical_p = 0.8,  top_k = 25, mirostat_mode = 0, tfs_z = 1.0,

	return response


for k in list(texts.keys()):
	print(f"reading file:{k}")
	
	print ('-------------------------')
	
	ques = texts[k]
	
	retrieved_docs = retriever.invoke(ques)
	
	answer = ask_rag(ques, retrieved_docs)
	#print(answer)


	with open(f'/home/data/LLM/LLM_outputs/medgemma/real_case/medgemma_8kxl/rag_on_trans_prompt6/{k}.txt' , 'w') as f:
		f.write(answer)
		
	

	save_retrieved(ques, retrieved_docs, k)

