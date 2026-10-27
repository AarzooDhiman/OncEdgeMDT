import camelot
import json
import os

# Paths are relative to the repository root, so the script can be run from anywhere.
from pathlib import Path as _P
REPO_ROOT = _P(__file__).resolve().parents[1]
RAG_DIR = REPO_ROOT / 'RAG'
CHUNK_DIR = RAG_DIR / 'chunks'  # chunk files written by chunking.py
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
file_name  = str(RAG_DIR / 'NICE_guidelines' / 'ng101.pdf')


'''TABLE_NAME = "Table 8 Benefits and risks of radiotherapy compared with no radiotherapy in the low-risk group described in recommendation 1.13.7"
SECTION_NAME = "1.13 Radiotherapy"
PAGE_NUMBER = '44'  # set dynamically if needed

tables = camelot.read_pdf(file_name, pages=PAGE_NUMBER, strip_text='\n')

all_chunks = []

for table_index, table in enumerate(tables):
    df = table.df
    # Remove completely empty rows
    df = df.dropna(how="all")
    headers = df.iloc[0].tolist()
    body = df.iloc[1:]
    
    for row_idx, row in body.iterrows():

        row_title = str(row.iloc[0]).strip()

        # Iterate through each column except first (row title column)
        for col_idx in range(1, len(headers)):

            column_name = str(headers[col_idx]).strip()
            cell_value = str(row.iloc[col_idx]).strip()

            # Skip empty cells
            if not cell_value:
                continue

            chunk = {
                "metadata": {
                    "recommendation_id": TABLE_NAME,
                    "section": SECTION_NAME,
                    "subsection": column_name,
                    "subsubsection": row_title,
                    "major": None,
                    "source": file_name,
                    "page": PAGE_NUMBER
                },
                "content": cell_value
            }

            all_chunks.append(chunk)
# Save to JSON
OUTPUT_PATH = str(CHUNK_DIR / "ng101_table8_chunks.json")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"Saved {len(all_chunks)} chunks to {OUTPUT_PATH}")'''

import os
import json

INPUT_FOLDER = str(CHUNK_DIR)
OUTPUT_FILE = str(RAG_DIR / "merged_output.json")

merged_data = []

for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".json"):
        file_path = os.path.join(INPUT_FOLDER, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            if isinstance(data, list):
                merged_data.extend(data)
            else:
                print(f"Skipping {filename} (not a list)")

# Save merged file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print(f"Merged {len(merged_data)} records into {OUTPUT_FILE}")
