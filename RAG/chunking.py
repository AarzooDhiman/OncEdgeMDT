from langchain_community.document_loaders import PyPDFLoader
import pypdf
import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
#import pdfplumber
from collections import defaultdict

paths = [
    "/home/data/LLM/NICE_guides/ng101.pdf",
    #"/home/data/LLM/NICE_guides/cg164.pdf",
    #"/home/data/LLM/NICE_guides/cg81.pdf"
]



docs = []
for p in paths:
    docs.extend(PyPDFLoader(p).load())
    
print (docs[:2])

print ('original data --------------------------------------------------------')
print (len(docs))

# Parse PDF and reconstruct hierarchy
# --------------------------------------------------


def is_toc_page(text):
    return (
        "contents" in text.lower()
        or text.count("....") > 10
        or sum(c.isdigit() for c in text) > 100
    )

docs = [d for d in docs if not is_toc_page(d.page_content)]

print ('removed contents page --------------------------------------------------------')

print (len(docs))

# --------------------------------------------
# STEP 3 — RESTORE NUMBERING WHEN BROKEN
# (.1 → previous_section.1)
# --------------------------------------------
def is_footer_or_header(line):
    l = line.lower().strip()
    #print ('inbside check for footer header')
    #print ('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    #print (l)
    return (
        # ---- generic NICE footers ----
        "nice" in l
        or "all rights reserved" in l
        or "notice-of-rights" in l
        or ("page" in l and any(ch.isdigit() for ch in l))
        or "©" in line

        # ---- document identifiers (CG / NG codes) ----
        or re.search(r"\((cg|ng)\d+\)", l)

        # ---- very long title-like header lines ----
        or ( "familial breast cancer: classification, care" in l
        )
        
        or ( r"page \d+ of" in l
        )
        or ( '50'== l
        )
    )
    
def restore_numbers(text):
    section_prefix = None
    lines = text.split("\n")
    fixed = []

    for line in lines:
        m = re.match(r"(\d+\.\d+(\.\d+)*)", line)
        if m:
            section_prefix = m.group(1)

        orphan = re.match(r"\.(\d+)", line)
        if orphan and section_prefix:
            line = section_prefix + "." + orphan.group(1)

        fixed.append(line)

    return "\n".join(fixed)
    
    
for d in docs[:10]:
    #print (d)
    #print ('--------------------------')
    d.page_content = restore_numbers(d.page_content)

# --------------------------------------------
# STEP 4 — DETECT SECTION HEADINGS
# --------------------------------------------
def classify_heading(line):
    line = line.strip()

    # Numbered section: 1.1 Diagnosis and assessment
    m = re.match(r"^(\d+\.\d+)\s+(.+)", line)
    if m:
        return ("section", m.group(2))

    # Major heading
    if line.lower() in ["recommendations", "introduction"]:
        return ("major", line)

    # Subsection heading (short, title-like)
    if (
        line
        and line[0].isupper()
        and not line.endswith(".")
        and not line.endswith("]")
        and not re.search(r"\d+\.\d+", line)
        and len(line) < 200
    ):
        return ("subsection", line)

    return (None, None)




    
# --------------------------------------------
# STEP 5 — SPLIT BY RECOMMENDATION ID
# --------------------------------------------
rec_pattern = r"(\d+\.\d+(\.\d+)*)"

def numbering_depth(num):
    return num.count(".") + 1


def is_context_heading(line):
    l = line.strip().lower()
    return l.startswith("update information")
    
def recommendation_complete(text_lines):
    """
    NICE recommendations always end with a closing ']'
    """
    return any("]" in line for line in text_lines)
        
def split_recommendations(doc, cur_sect, cur_subsect, cur_subsubsect,seen_rec, stop_after_context):

    seen_recommendation_in_subsection = seen_rec
    lines = doc.page_content.split("\n")
    #stop_after_context = False
    current_rec = None
    current_major = None
    current_section = cur_sect
    current_subsection = cur_subsect
    current_subsubsection = cur_subsubsect


    current_text = []

    chunks = []
    
    def flush(force=False):
        nonlocal current_rec, current_text
        
        if not current_rec:
            return
            
        if not force and not recommendation_complete(current_text):
            return
        
        #if current_rec:
        chunks.append(
                Document(
                    page_content=" ".join(current_text).strip(),
                    metadata={
                        "recommendation_id": current_rec,
                        "section": current_section,
                        "subsection": current_subsection,
                        "subsubsection": current_subsubsection,
                        "major": current_major,
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page"),
                    }
                )
            )
        current_rec = None
        current_text = []   

    #for line in lines:
    i = 0
    while i <len(lines):
        #print (i)
        line = lines[i].strip()
        
        if is_context_heading(line):
            flush(force=True)
            stop_after_context = True
            break
                
        
        #print ('===========================================')
        #print (line)
        #print ('------------------------------------------------')
        if not line or is_footer_or_header(line):
            #print ('found header-----------------------------------') 
            #print (line)
            i =i+1
            continue
            
        # ---- Merge multi-line headings ----
        if (
            line
            and line[0].isupper()
            and not line.endswith(".")
            and not re.search(r"\d+\.\d+", line)
        ):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if (
                    next_line
                    and next_line[0].islower()
                    and not next_line.endswith(".")
                    and not is_footer_or_header(next_line)
                ):
                    line = line + " " + next_line
                    i += 1
            
        # ---- SECTION HEADING (1.2 Something) ----
        m_section = re.match(r"^(\d+\.\d+)\s+(.+)", line)
        if m_section:
            flush(force=True)
            current_section = m_section.group(2)
            current_subsection = None
            current_subsubsection = None
            i=i+1
            continue

        # ---- HEADING DETECTION ----
        htype, htext = classify_heading(line)
        if htype == "subsection":

            flush(force=True)

            # If we already have a subsection,
            # this becomes a sub-subsection
            if seen_recommendation_in_subsection:
                current_subsection = htext
                current_subsubsection = None
                seen_recommendation_in_subsection = False
            else:
                # Otherwise it's a subsubsection
                current_subsubsection = htext
                

            i=i+1
            continue
            
        '''if htype and current_rec is None:
            if htype == "major":
                current_major = htext
                continue
            if htype == "section":
                current_section = htext
                current_subsection = None
                continue
            if htype == "subsection":
                current_subsection = htext
                continue'''

        # ---- RECOMMENDATION DETECTION ----
        m = re.search(r"\b(\d+\.\d+(\.\d+)*)\b", line)
        
        if m:
            print (m)
            num = m.group(1)

            # skip section numbers like 1.2
            if numbering_depth(num) < 3:
                i=i+1
                continue

            flush(force=True)
            current_rec = num
            current_text = [line[m.end():].strip()]
            seen_recommendation_in_subsection = True
        else:
            if current_rec:
                current_text.append(line)
        i=i+1
    flush(force=False)
    return chunks, current_section, current_subsection,current_subsubsection, seen_recommendation_in_subsection, stop_after_context




structured_chunks = []


cur_sect = None
cur_subsect = None
cur_subsubsect =None
stop_after_context =False

seen_rec = False
for d in docs:
    #print (d)
    if stop_after_context ==True:
        print("Context section found on page", doc.metadata.get("page"))
        break
    rec_chunks, cur_sect, cur_subsect, cur_subsubsect,seen_rec, stop_after_context = split_recommendations(d, cur_sect,cur_subsect, cur_subsubsect, seen_rec, stop_after_context )
    structured_chunks.extend(rec_chunks)



            
# --------------------------------------------
# STEP 6 — TOKEN WINDOW CHUNKING
# --------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=80
)

final_chunks = splitter.split_documents(structured_chunks)


import json

data = []

for doc in structured_chunks:
    data.append({
        "metadata": doc.metadata,
        "content": doc.page_content
    })

with open("/home/data/LLM/chunks/ng101_chunks.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("done")



'''for f in final_chunks:
    print (f)
    print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4')'''
    

print("Total chunks:", len(final_chunks))

from collections import defaultdict, Counter

# Collect all recommendation IDs
rec_ids = [doc.metadata["recommendation_id"] for doc in structured_chunks]

print("\n================ RECOMMENDATION VALIDATION ================\n")

# -----------------------------------------------------------
# 1️⃣ CHECK FOR DUPLICATES
# -----------------------------------------------------------
counts = Counter(rec_ids)
duplicates = [rid for rid, count in counts.items() if count > 1]

if duplicates:
    print("Duplicate recommendation IDs found:\n")
    for d in sorted(duplicates):
        print(f"  {d} (count = {counts[d]})")
else:
    print("No duplicate recommendation IDs found.\n")

# -----------------------------------------------------------
# 2️⃣ CHECK FOR MISSING NUMBERS PER PREFIX
# -----------------------------------------------------------
groups = defaultdict(list)

for rid in rec_ids:
    parts = rid.split(".")
    prefix = ".".join(parts[:-1])   # e.g. 1.7
    last = int(parts[-1])
    groups[prefix].append(last)

missing_found = False

for prefix, numbers in sorted(groups.items()):
    numbers = sorted(set(numbers))
    expected_range = range(numbers[0], numbers[-1] + 1)
    missing = sorted(set(expected_range) - set(numbers))

    if missing:
        missing_found = True
        print(f"\nMissing under {prefix}:")
        for m in missing:
            print(f"  {prefix}.{m}")

if not missing_found:
    print("No missing recommendation numbers detected.\n")

print("\n============================================================\n")
