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

    text = re.sub(r"^.*Journal of Clinical Oncology.*$\n?", '', text, flags=re.MULTILINE)
    text = re.sub(r"^.*NACT for Newly Diagnosed, Advanced Ovarian Cancer.*$\n?", '', text, flags=re.MULTILINE)
    text = re.sub(r"^.*University of South Florida.*$\n?", '', text, flags=re.MULTILINE)
    text = re.sub(r"^.*Copyright.*$\n?", '', text, flags=re.MULTILINE)
    text = re.sub(r"Table (\d+)", r" Table \1 ", text)
    text = re.sub(r"Figure (\d+)", r" Figure \1 ", text)
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

asco_path = "guidelineExtractors/guidelines/asco-guideline-update.pdf"
if os.path.exists(asco_path):
    try:
        reader = readpdf(asco_path)
    except FileNotFoundError as e:
        logger.error(f"File Not Found  error: {e}")
else:
    print(f"File Not Found Error: {asco_path}")
    exit(0)

# print(reader.pages[3].extract_text())
# preprocess(reader.pages[3].extract_text())
# creating a page object
# page = reader.pages[75]

# extracting text from page

invalid = [0, 1, 2, 7, 10, 20, 21, 22, 23, 24, 25, 26]

nccn=[]
for idx in range(len(reader.pages)):
    print(idx)
    if idx in invalid:
        continue
    text = preprocess(reader.pages[idx].extract_text()), 
    nccn.append({'page':idx,
    'text': text[0],})

output_path = "guidelineExtractors/guidelines/asco-guideline-update.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nccn, f, ensure_ascii=False, indent=2)

# print(f"Saved {len(nccn)} pages to {output_path}")
# print(reader.pages[12].extract_text())
# preprocess(reader.pages[72].extract_text())
# print(preprocess(reader.pages[12].extract_text()))