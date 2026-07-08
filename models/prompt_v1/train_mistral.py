import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth.chat_templates import standardize_data_formats, train_on_responses_only, get_chat_template
from sklearn.model_selection import train_test_split
from utils import PROMPT_IN_V1, PROMPT_OUT
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import EarlyStoppingCallback
#https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb


lr = 5e-5
num_epochs = 3
batch_size = 32
weight_decay= 1e-3
max_seq_length = 300 # Supports RoPE Scaling interally, so choose any!
seed=14
checkpoint   = "unsloth/Mistral-Small-3.2-24B-Instruct-2506-bnb-4bit"
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = checkpoint,
    max_seq_length = max_seq_length,   # Context length - can be longer, but uses more memory
    load_in_4bit = True,     # 4bit uses much less memory
    load_in_8bit = False,    # A bit more accurate, uses 2x memory
    full_finetuning = False, # We have full finetuning now!
)



df = pd.read_csv("../data_set.csv", sep="\t")

train, test = train_test_split(df, test_size=0.2, random_state=seed)
train, valid = train_test_split(train, test_size=0.1, random_state=seed)

#Text: é o texto original do autor
#generated_Text: é o texto generico (t)
dataset_train = Dataset.from_dict({"estilo_autoral": train["estilo_autoral"].tolist(), "estilo_academico": train["estilo_academico"].tolist(), "autor_alvo": train["user_id"].tolist()})
dataset_test = Dataset.from_dict({"estilo_autoral": test["estilo_autoral"].tolist(), "estilo_academico": test["estilo_academico"].tolist(), "autor_alvo": test["user_id"].tolist()})
dataset_valid = Dataset.from_dict({"estilo_autoral": valid["estilo_autoral"].tolist(), "estilo_academico": valid["estilo_academico"].tolist(), "autor_alvo": valid["user_id"].tolist()})


def generate_conversation(examples):
    textos_autores    = examples["estilo_autoral"]
    textos_academico  = examples["estilo_academico"]
    autores_alvo      = examples["autor_alvo"]

    conversations = []
    for texto_autor, texto_academico, autor_alvo in zip(textos_autores, textos_academico, autores_alvo):
        conversations.append([
            {"role" : "user",      "content" : PROMPT_IN_V1.format(autor_alvo=autor_alvo, input_text=texto_academico)},
            {"role" : "assistant", "content" : PROMPT_OUT.format(rewritten_text=texto_autor)},
        ])
    return { "conversations": conversations, }


reasoning_train = dataset_train.map(generate_conversation, batched = True)

reasoning_valid = dataset_valid.map(generate_conversation, batched = True)

def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
   return { "text" : texts, }

train_dataset = reasoning_train.map(formatting_prompts_func, batched = True)
valid_dataset = reasoning_valid.map(formatting_prompts_func, batched = True)




model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = lora_rank*2,  # Best to choose alpha = rank or rank*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = seed,
    use_rslora = False,   # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
)





trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = valid_dataset, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        per_device_eval_batch_size  = 1,
        gradient_accumulation_steps = 1, # Use GA to mimic batch size!
        warmup_steps = 5,
        eval_steps=0.1,
        logging_steps=0.1,
        run_name= "mistral-v1",
        output_dir="./mistral_finetuned",
        eval_strategy="steps",
        num_train_epochs = 3, # Set this for 1 full training run.
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = seed,
        save_steps=0.1,
        load_best_model_at_end=True,
        save_total_limit = 1,
        overwrite_output_dir = 'True',
        report_to = "none",
    ),
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)


trainer.train()
trainer.save_model("./mistral_finetuned")