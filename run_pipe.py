#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import itertools
import logging
import math
import os
import re
import datetime
import glob
import time
import pandas as pd
    
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset

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
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from knnlm import KNNWrapper, KNNSaver, KEY_TYPE, DIST
from retomaton import RetomatonWrapper

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.11.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

padding_index = -100

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    do_gen: bool = field(
        default=False,
        metadata={"help": "Generation"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list
    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    input_dir: Optional[str] = field(default=None, metadata={"help": "The input folder for pdf path report lookup (a text file)."})
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    stride: int = field(default=512)
    patience: int = field(default=None)
    prompt: str = field(default=None)

    report: bool = field(
        default=False, metadata={"help": "Input type Pathology report, task extract key info"}
    )
    process_findings: bool = field(
        default=False, metadata={"help": "Input type key information -> Processed info"}
    )
    restage: bool = field(
        default=False, metadata={"help": "Input type key information -> FIGO 2023 stage"}
    )
    recommend: bool = field(
        default=False, metadata={"help": "Input type FIGO 2023 stage -> treatment recommendation"}
    )

    def __post_init__(self):
        if self.train_file is None and self.input_dir is None:
            raise ValueError("Need either a dataset name or a folder.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class KNNArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    knn: bool = field(default=False)
    knn_gpu: bool = field(default=True)
    dstore_size: int = field(default=None, metadata={"help": "The size of the dstore."})
    knn_keytype: KEY_TYPE.from_string = field(default=KEY_TYPE.last_ffn_input)
    save_knnlm_dstore: bool = field(default=False)
    dstore_dir: str = field(default="checkpoints")
    knn_sim_func: DIST.from_string = field(default=DIST.l2)
    lmbda: float = field(default=0.25)
    k: int = field(default=1024)
    knn_temp: float = field(default=1.0)
    # Args for building the faiss index:
    build_index: bool = field(default=False)
    # faiss_index: str = field(default="checkpoints/index")
    ncentroids: int = field(default=4096)
    code_size: int = field(default=64)
    probe: int = field(default=32)
    num_keys_to_add_at_a_time: int = field(default=1000000)
    move_dstore_to_mem: bool = field(default=True)
    no_load_keys: bool = field(default=True)
    recompute_dists: bool = field(default=False)

    ## RetoMaton args:
    retomaton: bool = field(default=False)
    cluster_dstore: bool = field(default=False)
    no_pointer: bool = field(default=False)
    min_knns: int = field(default=1)
    max_knns: int = field(default=1024)
    num_clusters: int = field(default=500000)
    sample_size: int = field(default=20000000)
    members: str = field(default=None)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, KNNArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, knn_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, knn_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # logger.info(f"Training/evaluation parameters {training_args}")
    # logger.info(f"kNN parameters {knn_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    tokenizer.padding_side = "left"

    if model_args.model_name_or_path:
        model = GptOssForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            device_map= "auto",
            low_cpu_mem_usage=True,
            config=config,
            # quantization_config=Mxfp4Config(dequantize=True),
            # torch_dtype=torch.bfloat16,   
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # cfg = GptOssConfig.from_pretrained(model)
        # print(f"cfg: {cfg.quantization_config}")
    else:
        model = GptOssForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))
    logger.info(f'Model assigned to {model.device}')
    logger.info(model.hf_device_map)
    # for name, module in model.named_modules():
    #     print(name, ":", module)
    # print(model.base_model.layers)
    # last_layer_device = model.hf_device_map['model.layers.'+str(len(model.base_model.layers)-1)]
    # print(last_layer_device)

    # Injecting KNN
    dimension = model.config.hidden_size
    knn_wrapper = None
    knn_args.seed = training_args.seed

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    
    RELAY_KEY_INFO_NEW="""<|start|>system<|message|>You are an expert medical information extraction model specializing in gynecologic oncology.
Reasoning: low
Valid channels: analysis, final. Channel must be included for every message.<|end|>

<|start|>developer<|message|>Your task is to analyze uterine (endometrial) cancer pathology reports and extract only the specific pathologic and molecular features listed below.
You have already analyzed and extracted features from first few pages of the report.
Now, I will provide the remaining pages, which may contain new information or clarifications that modify, refine, or complete the existing extracted fields listed below.
Respond strictly in the given Markdown structure. Do not restate the report or add commentary.

Instructions:
First analyze the report to extract key information.
Focus on the diagnosis section when present.
Then analyze the previously extracted information and fill the gaps. 
Preserve existing correct values and update conflicting or more detailed information.
Never remove populated fields unless they are explicitly contradicted by the new text.
Report the information verbatim.
Use concise medical phrasing exactly as written in the report when possible.
Mention any key information not already captured at the end with the heading "Other important information".
Maintain the same headings and indentation style.
<|end|>

<|start|>user<|message|>**Input Pathology Report:**
---
{report}
---

"""

    KEY_INFO="""<|start|>system<|message|>You are an expert medical information extraction model specializing in gynecologic oncology.
Reasoning: low
Valid channels: analysis, final. Channel must be included for every message.<|end|>

<|start|>developer<|message|>Your task is to analyze uterine (endometrial) cancer pathology reports. Identify the diagnosis section and extract only the specific pathologic and molecular features listed below.
Respond strictly in the given Markdown structure.

Instructions:
Focus on the diagnosis section.
Fill every heading and subheading with the specific information mentioned in the report.
If a detail is not mentioned, write “Not reported.”
Report the information verbatim.
Use concise medical phrasing exactly as written in the report when possible.
Mention any key information not already captured at the end with the heading "Other important information".
Maintain the same headings and indentation style.

Output Format
---
### Key Findings from Pathology Report

**Histological Type:**  
**Histological Grade(FIGO):**  

**Myometrial Invasion status:**   

**Lymphovascular Space Invasion (LVSI) Status:**  
**Cervical Stromal Invasion Status:**  
**Adnexal Involvement Status:**  
**Uterine Serosal Involvement Status:**  

**Lymph Node Metastasis:**  
- Pelvic:   
- Pariaortic:  
- Extra-abdominal:
- Intra-abdominal:


**Molecular Classification:**  
- DNA polymerase epsilon (POLE) mutated:
- Mismatch repair deficient:
- Copy number high/p53 abnormal:
- Copy number low/no specific molecular profile.

Metastasis:
- Vagina:
- Parametria:
- Pelvic Peritoneal metastasis:
- Extrapelvic Peritoneal Metastasis:
- Bladder Mucosa:
- Intestinal/Bowel Mucosa:
- Lungs:
- Liver:
- Brain:
- Bone:

Other important information:
---
<|end|>

<|start|>user<|message|>**Input Pathology Report:**
---
{report}
---
<|end|>
"""

    IMP_FINDINGS = '''<|start|>system<|message|>You are a clinical reasoning model specialized in gynecologic oncology. Under 'Histological Type' along with the reported type mention if it's non-aggressive or aggressive type in paranthesis.
Knowledge cutoff: 2026-2023
Current date: {cur_date}
Reasoning: medium
Valid channels: analysis, final. Channel must be included for every message.<|end|>

<|start|>developer<|message|>Given key uterine cancer features from a pathology report, retain the important details and ommit unnecessary information.

1. Maintain the same headings and indentation style.
2. Fields that are not reported can be ommitted.
3. State the valid fields verbatim and do not add any commentary or interpretation.
4. Maintain the same headings and indentation style.
5. Do not rewrite the reported name but append the aggressiveness.
5. Non-aggressive histological types are composed of low-grade (grades 1 and 2) Endometrioid carcinoma (EEC).
6. Aggressive histological types are composed of high-grade EECs(grade 3), serous carcinoma (SC), clear cell carcinoma (CCC), mixed carcinoma (MC), undifferenti-ated carcinoma (UC), carcinosarcoma (CS), other unusual types,such as mesonephric-like and gastrointestinal mucinous type car-cinomas
7. Metastasis can be through direct extension, lymphatic or transcoelomic spread. The mode of spread does not need to be specified but the involved site should be mentioned under the appropriate heading.
8. Omental involvement implies extrapelvic peritoneal metastasis and should be mentioned under respective metastasis heading.


<|end|>

<|start|>user<|message|>
{key_features}
<|end|>
'''

    STAGING="""<|start|>system<|message|>You are a clinical reasoning model specialized in gynecologic oncology.
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
---

{key_features}

---
<|end|>
<|start|>assistant<|message|>"""

    DISCLAIMER = "⚠️ This tool is for educational purposes only. All staging decisions must be confirmed by a qualified gynecologic oncologist using the full FIGO 2023 guidelines"

    def word_count(text):
        return len(text.split())


    def format_extraction(report, max_words=1200):
        # print(word_count(report))

        if word_count(report) <= max_words:
            return [KEY_INFO.format(report=report)]

        lines = report.splitlines()

        parts = []
        current_part = []
        current_wc = 0

        for line in lines:
            lwc = word_count(line)

            # if adding this line exceeds limit → finalize current chunk
            if current_wc + lwc > max_words and current_part:
                parts.append("\n".join(current_part))
                current_part = [line]
                current_wc = lwc
            else:
                current_part.append(line)
                current_wc += lwc

        # add final chunk
        if current_part:
            parts.append("\n".join(current_part))

        formatted_reports = []
        for i, part in enumerate(parts):
            if i == 0:
                formatted_reports.append(KEY_INFO.format(report=part))
            else:
                formatted_reports.append(RELAY_KEY_INFO_NEW.format(report=part))

        return formatted_reports

    def load_keyinfo(key_info):
        report = ""
        key_info = re.sub(r"\n```File Name: TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}```\n\n?", '', key_info, flags=re.MULTILINE)
        if data_args.process_findings:
            report= IMP_FINDINGS.format(cur_date = time.strftime('%Y-%m-%d'), key_features = key_info)
        else:
            report= STAGING.format(cur_date = time.strftime('%Y-%m-%d'), key_features = key_info)
        return report

    if knn_args.retomaton or knn_args.cluster_dstore:
        knn_wrapper = RetomatonWrapper(dstore_size=knn_args.dstore_size, dstore_dir=knn_args.dstore_dir, 
            dimension=dimension, 
            knn_sim_func=knn_args.knn_sim_func, knn_keytype=knn_args.knn_keytype,
            no_load_keys=knn_args.no_load_keys, move_dstore_to_mem=knn_args.move_dstore_to_mem, knn_gpu=knn_args.knn_gpu,
            recompute_dists=knn_args.recompute_dists,
            k=knn_args.k, lmbda=knn_args.lmbda, knn_temp=knn_args.knn_temp, probe=knn_args.probe,
            no_pointer=knn_args.no_pointer, min_knns=knn_args.min_knns, max_knns=knn_args.max_knns,
            members=knn_args.members)
    elif knn_args.knn:
        knn_wrapper = KNNWrapper(dstore_size=knn_args.dstore_size, dstore_dir=knn_args.dstore_dir, 
            dimension= dimension, 
            knn_sim_func=knn_args.knn_sim_func, knn_keytype=knn_args.knn_keytype,
            no_load_keys=knn_args.no_load_keys, move_dstore_to_mem=knn_args.move_dstore_to_mem, knn_gpu=knn_args.knn_gpu,
            recompute_dists=knn_args.recompute_dists,
            k=knn_args.k, lmbda=knn_args.lmbda, knn_temp=knn_args.knn_temp, probe=knn_args.probe)
    elif knn_args.save_knnlm_dstore or knn_args.build_index:
        knn_wrapper = KNNSaver(dstore_size=knn_args.dstore_size, dstore_dir=knn_args.dstore_dir, 
            dimension=dimension, knn_keytype=knn_args.knn_keytype)
    
    if knn_wrapper is not None:
        knn_wrapper.break_into(model)
        if knn_args.retomaton:
            if model_args.do_gen:
                knn_wrapper.do_gen = True
            knn_wrapper.load_retomaton()

    
    def parse_stage(actual, predicted):
        if not predicted:
            return 0

        match = re.match(r'^(I{1,3}|IV)([A-Z]+)([0-9]*)', predicted)
        if match:
            main = match.group(1)   # I, II, III, IV
            subtype = match.group(2)  # A, B, C
            number = match.group(3)   # 1, 2, 3,
        else:
            return 0
        match2 = re.match(r'^(I{1,3}|IV)([A-Z]+)([0-9]*)', actual)
        if match2:
            main2 = match2.group(1)   # I, II, III, IV
            subtype2 = match2.group(2)  # A, B, C
            number2 = match2.group(3)   # 1, 2,
        if main and main2:
            if main == main2:
                if subtype == subtype2:
                    return 0.66
                else:
                    return 0.33
        return 0

    def write_result(root, output, filename, idx=0, actual_stage=None):
        mode = 'w'
        if not os.path.exists(root):
            os.makedirs(root)
        if knn_args.retomaton:
            dir = root + f"/reto{knn_args.dstore_size}_{knn_args.lmbda}"
            if not os.path.exists(dir):
                os.makedirs(dir)
        else:
            dir = root + "/lm"
            if not os.path.exists(dir):
                os.makedirs(dir)
        if data_args.report:
            with open(root + f"/key_info.md", mode) as file:
                file.write(f"\n\n```File Name: {filename}```\n\n")
                file.write(output.split('assistantfinal')[-1])
            with open(dir + f"/{filename}_findings_cot.txt", 'w') as file:
                file.write(output.split('assistantfinal')[0].split('<|channel|>analysis')[-1])
        if data_args.process_findings:
            if idx == 0:
                with open(root + f"/processed_info.md", mode) as file:
                    file.write(f"\n\n```File Name: {filename}```\n\n")
                    file.write(output.split('assistantfinal')[-1])
            else:
                with open(root + f"/processed_info{idx}.md", mode) as file:
                    file.write(f"\n\n```File Name: {filename}```\n\n")
                    file.write(output.split('assistantfinal')[-1])
            with open(dir + f"/{filename}_findings_processed_cot.txt", 'w') as file:
                file.write(output.split('assistantfinal')[0].split('<|channel|>analysis')[-1])
        if data_args.restage:
            with open(dir + f"/restage.md", mode) as file:
                file.write(f"\n\n```File Name: {filename}```\n\n")
                file.write(output.split('assistantfinal')[-1])
            with open(dir + f"/{filename}_restage_cot.txt", 'a') as file:
                file.write(output.split('assistantfinal')[0].split('<|start|>assistant<|message|>')[-1])
            to_path = f"reto{knn_args.dstore_size}_{knn_args.lmbda}_results.csv" if knn_args.retomaton else f"restage_results.csv"
            print(f"logging results to {to_path}")
            if os.path.exists(to_path):
                mode = 'a'
            with open(to_path, mode=mode, newline='', encoding='utf-8') as csv_file:
                # print("writing output")
                writer = csv.DictWriter(csv_file, fieldnames=['File', 'Actual stage', 'RetoMaton stage', 'Accuracy'])
                mapping = {'1':'I', '2' : 'II', '3' : 'III', '4' : 'IV'}
                actual_stage = mapping.get(actual_stage[0], actual_stage[0]) + actual_stage[1:]
                # print(f"actual stage: {actual_stage}")
                if mode == 'w':
                    writer.writeheader()
                output = output.replace('*', '')
                match = re.search(r'FIGO2023Stage\s*:\s*([A-Za-z0-9]+)', output)
                stage = 'UND'
                if match:
                    stage = match.group(1)
                accuracy = 1 if stage == actual_stage else parse_stage(actual_stage, stage)
                print(f"actual stage: {actual_stage}, predicted stage: {stage}, accuracy: {accuracy}")
                writer.writerow({'File': filename, 'Actual stage': actual_stage, 'RetoMaton stage': stage, 'Accuracy': accuracy})
            

    def find_files(root_directory, ext):
        file_list = []

        # Validate root directory exists
        if not os.path.exists(root_directory):
            raise FileNotFoundError(f"Directory not found: {root_directory}")

        print(f"Scanning directory: {root_directory}")

        # os.walk recursively yields (dirpath, dirnames, filenames) for each directory
        # This is more efficient than recursive function calls for deep hierarchies
        for dirpath, dirnames, filenames in os.walk(root_directory):
            for filename in filenames:
                # Case-insensitive TXT detection to handle .TXT, .txt, .Txt, etc.
                if filename.lower().endswith(ext):
                    if ('lm' in dirpath or 'reto' in dirpath) and ext == ".txt":
                        continue
                    txt_path = os.path.join(dirpath, filename)
                    file_list.append(txt_path)

        print(f"Found {len(file_list)} text file(s) to process")
        return file_list

    time = datetime.datetime.now()
    
    if model_args.do_gen is not None:
        batch_size = 1
        num_beams = 5
        outputs = []
        tokenizer.padding_side='left'
        
        text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        
        retry_idx = []
        if data_args.report:
            ext = ".txt"
        elif data_args.process_findings:
            ext = "key_info.md"
        elif data_args.restage:
            ext = "processed_info.md"

        if data_args.input_dir is not None:
            files = find_files(root_directory = data_args.input_dir, ext = ext)
            num_files = len(files)
        elif data_args.train_file is not None:
            file_df = pd.read_csv(data_args.train_file, header='infer')
            # file_df = file_df[~file_df['2023 FIGO stage postop'].str.contains('m', na=False)].reset_index(drop=True)
            files = file_df['path'].tolist()
            list = []
            for idx in [18, 27, 31, 92, 109]:
                file =  files[idx]   
                if not isinstance(file, str):
                    continue
                f = find_files('/data/rushitha/cancer/qwen-ocr/'+file, ext)
                if len(f)==0:
                    continue
                list.append(f[0])
            files = list
            logger.info(f"Files list: {files}")
            num_files = len(files)

        for idx, file in enumerate(files):
            read_file = open(file, 'r').read()
            file_path = os.path.dirname(file)
            match = re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", file_path)
            file_name = match.group(0) if match else f"file_{idx}"
            print(f"Processing file: {file_name}")
            start = datetime.datetime.now()
            try:
                if data_args.report:
                    # knn_wrapper.break_out()
                    logger.info(f"Extracting key info from {idx+1}/{num_files}: {file}.")
                    max_new_tokens = 2048 + 1024
                    input = format_extraction(read_file)
                    output = text_generator(input[0], 
                        batch_size = batch_size, 
                        max_new_tokens=max_new_tokens,
                        num_beams = num_beams, 
                        temperature = 0.3,
                        top_p = 0.9,
                        repetition_penalty = 1.5,
                        early_stopping =  True,
                    )
                    if 'assistantfinal' not in output[0]['generated_text']:
                        logger.error("Extraction incomplete")
                        final_key_info = output[0]['generated_text']
                        continue
                    id = 0 if len(input) == 1 else 1
                    write_result(file_path, output[0]['generated_text'], file_name, id) 
                    if len(input) > 1:
                        for i in range(1,len(input)):
                            key_info = output[0]['generated_text'].split('assistantfinal')[-1]
                            logger.info(f"Intermediary findings\n {key_info}")
                            key_info = key_info.replace("Key Findings from Pathology Report", "Key Findings from starting pages of Pathology Report")
                            curr_input = input[i] + key_info + '\n---\n<|end|>'
                            output = text_generator(curr_input, 
                                batch_size = batch_size, 
                                max_new_tokens=max_new_tokens, 
                                num_beams = num_beams, 
                                temperature = 0.3,
                                top_p = 0.9,
                                repetition_penalty = 1.5,
                                early_stopping =  True,
                            )
                            if 'assistantfinal' not in output[0]['generated_text']:
                                logger.error(f"{output[0]['generated_text']}\nExtraction incomplete")
                                continue
                            if i < len(input)-1: write_result(file_path, output[0]['generated_text'], file_name, i+1) 
                    final_key_info = output[0]['generated_text'].split('assistantfinal')[-1]
                    # logger.info(f"Final findings\n {final_key_info}")
                    final_key_info = final_key_info.replace("Key Findings from starting pages of Pathology Report", "Key Findings from Pathology Report")
                    write_result(file_path, output[0]['generated_text'], file_name, 0)
                    if data_args.process_findings:
                        process_input = IMP_FINDINGS.format(cur_date = time.strftime('%Y-%m-%d'), key_features = final_key_info) 
                if data_args.process_findings:# or data_args.report:
                    logger.info(f"Processing key info: {idx+1}/{num_files}: {file}.")
                    max_new_tokens = 2048 + 1048
                    key_info = load_keyinfo(read_file)
                    print(f"key info: {len(key_info.split())}")
                    if len(key_info.split()) > 700:
                        continue
                    # print(f"key info: {key_info}")
                    output = text_generator(key_info, 
                        batch_size = batch_size, 
                        max_new_tokens=max_new_tokens, 
                        num_beams = num_beams, 
                        temperature = 0.3,
                        top_p = 0.9,
                        repetition_penalty = 2.0,
                        early_stopping =  True,
                    )
                    if 'assistantfinal' not in output[0]['generated_text']:
                        logger.error(f"Extraction incomplete")
                        # continue
                    restaged = output[0]['generated_text']
                    logger.info(f"Restaged\n {restaged}")
                    
                    write_result(file_path, restaged, file_name)
                    
                if data_args.restage:# or data_args.report:
                    logger.info(f"Restaging from key info {idx+1}/{num_files}: {file}.")
                    max_new_tokens = 2048
                    key_info = load_keyinfo(read_file)
                    output = text_generator(key_info, 
                        batch_size = batch_size, 
                        max_new_tokens=max_new_tokens,
                        num_beams = num_beams, 
                        temperature = 0.3,
                        top_p = 0.9,
                        repetition_penalty = 2.0,
                        early_stopping =  True,
                    )
                    if 'assistantfinal' not in output[0]['generated_text']:
                        logger.error(f"Extraction incomplete")
                        # continue
                    restaged = output[0]['generated_text']
                    logger.info(f"Restaged\n {restaged}")
                    # path = f"{training_args.output_dir}{raw_datasets[data_args.eval_subset][idx]['Numerical identifer']}/restaged_{time.strftime('%Y-%m-%d_%H:%M')}"
                    actual_stage = file_df[file_df['Numerical identifer'] == file_name]['2023 FIGO stage postop'].values[0]
                    print(f"actual stage: {actual_stage}")
                    write_result(file_path, restaged, file_name, None, actual_stage)
                    # recommendation_input = STAGING.format(key_features = final_key_info)
                if data_args.recommend:
                    logger.info(f"Generating treatment recommendation for the pathology report. Name: {raw_datasets[data_args.eval_subset][idx]['Numerical identifer']}.")
                    output = text_generator(recommendation_input, batch_size = batch_size, max_new_tokens=2048, num_beams = num_beams)
                    if 'assistantfinal' not in output[0]['generated_text']:
                        logger.error(f"Extraction incomplete")
                        continue
                    treatment = output[0]['generated_text'].split('assistantfinal')[-1]
                    logger.info(f"Treatment\n {treatment}")
                    path = f"{training_args.output_dir}/{raw_datasets[data_args.eval_subset][idx]['Numerical identifer']}/treatment_{time.strftime('%Y-%m-%d_%H:%M')}"
                    write_result(path, treatment, raw_datasets[data_args.eval_subset][idx])
            except Exception as e:
                raise e
                continue
            logger.info(f"That took {datetime.datetime.now()-start}s")
            
    if knn_args.build_index:
        knn_wrapper.build_index()

    if knn_args.cluster_dstore:
        knn_wrapper.cluster_dstore(num_clusters=knn_args.num_clusters, sample_size=knn_args.sample_size, model=model)
    
    if knn_wrapper is not None:
        knn_wrapper.break_out()


if __name__ == "__main__":
    main()
