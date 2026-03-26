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

    text = re.sub(r"^.*Library.*$\n?", '', text, flags=re.MULTILINE)
    text = re.sub(r"^.*BEREK et al..*$\n?", '', text, flags=re.MULTILINE)
    text = re.sub(r"^.*University of South Florida.*$\n?", '', text, flags=re.MULTILINE)
    # text = re.sub(r"^.*Copyright.*$\n?", '', text, flags=re.MULTILINE)
    text = re.sub(r"\d{3}\s+\|\s+\n", "", text)
    text = re.sub(r"\s+\|\s+\d{3}\n", "", text)
    # text = re.sub(r"Figure (\d+)", r" Figure \1 ", text)
    # print(text)
    # title = text.split('\n')[0]
    # pattern = re.compile(r'(?m)^.*».*$') # \u00BB for »
    # headings = pattern.findall(text)
    # if headings:
    #     title = headings[-1]
        
    # m = re.search(rf'{re.escape(title)}', text)
    # if m:
    #     text = text[m.end()+1:]
    # else:
    #     print("No match")
    
    # 1) Replace “single” newlines (not part of a double-newline) with a space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # 2) Optionally strip extra spaces
    text = re.sub(r' {2,}', ' ', text)
    # title = re.sub(r' {2,}', ' ', title)
    # print(text)
    return text

asco_path = "guidelines/FIGO_staging_of_endometrial_cancer_2023.pdf"
if os.path.exists(asco_path):
    try:
        reader = readpdf(asco_path)
    except FileNotFoundError as e:
        logger.error(f"File Not Found  error: {e}")
else:
    print(f"File Not Found Error: {asco_path}")
    exit(0)

# print(reader.pages[1].extract_text())
# print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# print(preprocess(reader.pages[1].extract_text()))
# preprocess(reader.pages[3].extract_text())
# creating a page object
# page = reader.pages[0]

# extracting text from page

invalid = [2,4]

json_file=[]
idx = 2
json_file.append({'page':0,
'text': preprocess(reader.pages[idx].extract_text()),})

for idx in range(len(reader.pages)):
    # print(idx)
    if idx not in invalid:
        continue
    text = preprocess(reader.pages[idx].extract_text()), 
    json_file.append({'page':idx,
    'text': text[0],})

output_path = "guidelines/only_figo_staging.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(json_file, f, ensure_ascii=False, indent=2)

# print(f"Saved {len(nccn)} pages to {output_path}")
# print(reader.pages[12].extract_text())
# preprocess(reader.pages[72].extract_text())
# print(preprocess(reader.pages[12].extract_text()))