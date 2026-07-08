from gc import enable
from unsloth import FastLanguageModel
from transformers import TextStreamer
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
from utils import PROMPT_IN_V2

load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
seed=14

max_seq_length= 1024


model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "./mistral_finetuned/checkpoint-912/", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 300,
        load_in_4bit = load_in_4bit,
        fast_inference = False
    )




df = pd.read_csv("../data_set.csv", sep="\t")

train, _ = train_test_split(df, test_size=0.2, random_state=seed)
_, valid = train_test_split(train, test_size=0.1, random_state=seed)

dataset_valid = Dataset.from_dict({"estilo_autoral": valid["estilo_autoral"].tolist(), "estilo_academico": valid["estilo_academico"].tolist(), "autor_alvo": valid["user_id"].tolist()})



def generate_conversation(examples):
    textos_autores    = examples["estilo_autoral"]
    textos_academico  = examples["estilo_academico"]
    autores_alvo      = examples["autor_alvo"]

    conversations = []
    for texto_autor, texto_academico, autor_alvo in zip(textos_autores, textos_academico, autores_alvo):
        conversations.append([
            {"role" : "user",      "content" : PROMPT_IN_V2.format(input_text=texto_academico)}
        ])
    return { "conversations": conversations, }


reasoning_valid = dataset_valid.map(generate_conversation, batched = True)

texts = [tokenizer.apply_chat_template(ele, tokenize = False, add_generation_prompt = True) for ele in reasoning_valid["conversations"]]


generated = []
for ele in tqdm(texts):
    generated_ids = model.generate(**tokenizer(ele, return_tensors = "pt").to("cuda"),     temperature = 0.8,
    top_p = 0.95, max_new_tokens = 1024)
    generated.append(tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip())

def write_file(data):
    
    with open("./preds_mistral.txt", "w+", encoding="utf-8") as outfile:
        outfile.write("\n".join(data))

write_file(generated)


