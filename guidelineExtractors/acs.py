from pypdf import PdfReader
import os
import pikepdf
import re, json

def readpdf(file):
    reader = PdfReader(file)
    if reader.is_encrypted:
        pdf = pikepdf.open(file, password="", allow_overwriting_input=True)
        pdf.save(file)
        reader = PdfReader(file)
    return reader


def preprocess(text):
    text = text.replace('’','\'')

    text = re.sub(r"^cancer\.org \| \d{1}\.\d{3}\.\d{3}\.\d{4}.*$\n?", '', text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s*", '', text, flags=re.MULTILINE)
    text = text.replace("\u00A0", " ")              # Non-breaking space character (NBSP)
    
    text = ' '.join(text.split('\n')[1:])           # Ignore page number
    text = re.sub(r' {2,}', ' ', text)              # Strip extra spaces
    
    return text

path = "guidelineExtractors/guidelines/"
file = "acs.pdf"

file_path = path + file

if os.path.exists(file_path):
    try:
        reader = readpdf(file_path)
    except FileNotFoundError as e:
        logger.error(f"File Not Found  error: {e}")
else:
    print(f"File Not Found Error: {file_path}")
    exit(0)


invalid = [11]

nccn=[]
for idx in range(len(reader.pages)):
    if idx in invalid:
        continue
    text = preprocess(reader.pages[idx].extract_text()), 
    nccn.append({'page': idx+1,
    'text': text[0],})

output_path = path + file.split('.')[0] + '.json'
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nccn, f, ensure_ascii=False, indent=2)
