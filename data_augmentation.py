import datasets
from datasets import load_dataset
from tqdm import tqdm
import csv
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    Mxfp4Config, 
    GptOssForCausalLM,
    AutoModelForCausalLM,
    GptOssConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    pipeline,
)
from datasets import load_from_disk

data_files = {}
data_files["train"] = '/data/rushitha/cancer/knn-transformers/basecaseRestaged.csv'
raw_datasets = load_dataset('csv',data_files = data_files )

def format_prompt(report):
        prompt = """<|start|>system<|message|>You are a clinical reasoning model specialized in gynecologic oncology.
Knowledge cutoff: 2026-2023
Current date: {cur_date}
Reasoning: medium
Valid channels: analysis, final. Channel must be included for every message.<|end|>

<|start|>developer<|message|>Given key uterine cancer features from a pathology report, determine the FIGO 2023 Uterine Cancer Stage.
Respond strictly in the given Markdown structure. Do not restate the features or add commentary.

Let's recall FIGO 2023 staging for endometrial carcinoma. Stage I Confined to the uterine corpus and ovary. Stage IA Disease limited to the endometrium OR non-aggressive histological type, i.e. low-grade endometroid, with invasion of less than half of myometrium with no or focal lymphovascular space involvement (LVSI) OR good prognosis disease. Stage IA1 Non-aggressive histological type limited to an endometrial polyp OR confined to the endometrium. Stage IA2 Non-aggressive histological types involving less than half of the myometrium with no or focal LVSI. Stage IA3 Low-grade endometrioid carcinomas limited to the uterus and ovary. Stage IB Non-aggressive histological types with invasion of half or more of the myometrium, and with no or focal LVSI. Stage IC Aggressive histological types limited to a polyp or confined to the endometrium. Stage II Invasion of cervical stroma without extrauterine  =extension OR with substantial LVSI OR aggressive histological types with myometrial invasion. Stage IIA Invasion of the cervical stroma of non-aggressive histological types. Stage IIB Substantial LVSI of non-aggressive histological types. Stage IIC Aggressive histological types with any myometrial involvement. Stage III Local and/or regional spread of the tumor of any histological subtype. Stage IIIA Invasion of uterine serosa, adnexa, or both by direct extension or metastasis. Stage IIIA1 Spread to ovary or fallopian tube (except when meeting stage IA3 criteria). Stage IIIA2 Involvement of uterine subserosa or spread through the uterine serosa. Stage IIIB Metastasis or direct spread to the vagina and/or to the parametria or pelvic peritoneum. Stage IIIB1 Metastasis or direct spread to the vagina and/or the parametria. Stage IIIB2 Metastasis to the pelvic peritoneum. Stage IIIC Metastasis to the pelvic or para-aortic lymph nodes or both. Stage IIIC1 Metastasis to the pelvic lymph nodes. Stage IIIC1i Micrometastasis. Stage IIIC1ii Macrometastasis. Stage IIIC2 Metastasis to para-aortic lymph nodes up to the renal vessels, with or without metastasis to the pelvic lymph nodes. Stage IIIC2i Micrometastasis. Stage IIIC2ii Macrometastasis. Stage IV Spread to the bladder mucosa and/or intestinal mucosa and/or distance metastasis. Stage IVA Invasion of the bladder mucosa and/or the intestinal/bowel mucosa. Stage IVB Abdominal peritoneal metastasis beyond the pelvis. Stage IVC Distant metastasis, including metastasis to any extra- or intra-abdominal lymph nodes above the renal vessels, lungs, liver, brain, or bone
FIGO endometrial cancer stage with molecular classification. Stage designation Molecular findings in patients with early endometrial cancer (Stages I and II after surgical staging) Stage IAm_POLEmut: POLEmut endometrial carcinoma, confined to the uterine corpus or with cervical extension, regardless of the degree of LVSI or histological type. Stage IICm_p53abn: p53abn endometrial carcinoma confined to the uterine corpus with any myometrial invasion, with orwithout cervical invasion, and regardless of the degree of LVSI or histological type.
FIGO Stages I and II are based on surgical/anatomical and histological findings. In case the molecular classification reveals POLEmut or p53abn status, the FIGO stage is modified in the early stage of the disease. This is depicted in the FIGO stage by the addition of "m" for molecularclassification, and a subscript is added to denote POLEmut or p53abn status, as shown below. MMRd or NSMP status do not modify early FIGOstages; however, these molecular classifications should be recorded for the purpose of data collection. When molecular classification revealsMMRd or NSMP, it should be recorded as Stage Im_MMRd or Stage Im_NSMP and Stage IIm_MMRd or Stage IIm_NSMP.

Instructions:
Analyze the key features provided.
Use the official 2023 FIGO staging guidelines that combine anatomic extent and molecular subtype (POLEmut, p53abn, MMRd, NSMP).
Output the final stage notation exactly as defined in FIGO 2023 (e.g., IA2, IIIC1, IVB, IAm_POLEmut, IICm_p53abn) and specify grade separately.
If the stage qulifies for multiple stages/substages pick the advanced stage.
Do not derrive molecular classification unless specified.
Use concise medical phrasing when possible.
Maintain the same headings and indentation style.

Output Format

**FIGO2023Stage**: ,
**TumorGrade**: ,
**MolecularModifierApplied**: ,
**MolecularSubtype**: ,
**Rationale**: "Concise explanation of how key features determine stage and grade",
**MissingInformation**: ["list any missing or ambiguous fields"]

---

<|end|>
<|start|>user<|message|>The key features of the patholgy report are:

"""
        for key in report.keys():
            if key == 'Grade' or key == 'check' or key == 'restaged':
                continue
            prompt+=f"**{key}**: {report[key]}\n"
        prompt+="""

---
<|end|>
<|start|>assistant<|message|>"""

        report['input_prompt'] = prompt
        return report

raw_datasets = raw_datasets.map(
    format_prompt
)

# print(raw_datasets['train'][0]['input_prompt'])

# raw_datasets['train'].to_csv('/data/rushitha/cancer/knn-transformers/basecase032326.csv')

model_name_or_path = "openai/gpt-oss-20b"
config = AutoConfig.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = GptOssForCausalLM.from_pretrained(
    model_name_or_path,
    device_map= "auto",
    low_cpu_mem_usage=True,
    config=config,
)
model.resize_token_embeddings(len(tokenizer))


batch_size = 1
num_beams = 5
tokenizer.padding_side='left'

text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generte_output(example):
    example['output'] = text_generator(example['input_prompt'], 
        batch_size = batch_size, 
        max_new_tokens=4196,
        num_beams = num_beams, 
        temperature = 0.3,
        top_p = 0.9,
        repetition_penalty = 1.5,
        early_stopping =  True,
    )
    return example

raw_datasets = raw_datasets.map(generte_output, desc="Generating Prompt",)
raw_datasets['train'].to_csv('/data/rushitha/cancer/knn-transformers/basecaseDstore032326.csv')