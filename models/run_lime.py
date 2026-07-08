

from lime.lime_text import LimeTextExplainer
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
seed = 14

df = pd.read_csv("./data_set.csv", sep="\t")



train, test = train_test_split(df, test_size=0.2, random_state=seed)

test.reset_index(inplace=True)
test.drop(columns=["index"], inplace=True)


df_qwen     = pd.read_csv("./testes/preds_qwen.txt", sep="\t", header=None).rename(columns={0: "qwen_outs"}, inplace=False)
df_final = pd.concat([df_qwen, test], axis=1)

def compute_acc(candidates):

    model     = AutoModelForSequenceClassification.from_pretrained("../dataset/data/ustance/classification/results/checkpoint-520/")
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    model = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=1)
    preds = []
    for pred in model(candidates, truncation=True, padding=True, max_length=512):
        preds.append(int(pred[0]["label"].split("LABEL_")[1]))
    
    return preds

df_final["preds"] = compute_acc(df_final["qwen_outs"].tolist())



labels = {  'r2_lu_876': 0,
            'r2_cl_2165': 1,
            'r2_gl_1071': 2,
            'r2_bo_344': 3,
            'r2_bo_208': 4,
            'r2_lu_94': 5}

"""convert columns user_id with the dicitonary labels"""
df_final["label"] = df_final["user_id"].apply(lambda x: labels[x] if x in labels else -1)


df_final["iscorrect"]  = df_final["preds"] == df_final["label"]
# Função para extrair um exemplo de acerto e um de erro por usuário
def extrair_acertos_erros_por_usuario(df):
    exemplos = []

    for user_id in df['user_id'].unique():
        df_user = df[df['user_id'] == user_id]
        # Um exemplo incorreto (label == 0)
        erros = df_user[df_user['iscorrect'] == False]
        if not erros.empty:
            exemplo_erro = erros.sample(1, random_state=seed)
            exemplos.append(exemplo_erro)

    resultado = pd.concat(exemplos).reset_index(drop=True)
    return resultado

df_to_be_analised = extrair_acertos_erros_por_usuario(df_final)


model     = AutoModelForSequenceClassification.from_pretrained("../dataset/data/ustance/classification/results/checkpoint-520/")
tokenizer = AutoTokenizer.from_pretrained("pablocosta/bertabaporu-base-uncased")

# Pipeline para predição
def predict_prob(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs.detach().numpy()




# Inicializa LIME
class_names = ["r2_lu_876", "r2_cl_2165", "r2_gl_1071", "r2_bo_344", "r2_bo_208", "r2_lu_94"]  # Substitua pelos nomes reais das classes
explainer = LimeTextExplainer(class_names=class_names)

# Textos simulados
textos_gerados = df_to_be_analised["qwen_outs"].tolist()
print(textos_gerados)
input()
# Coleta explicações simuladas
tabela_resultado = []

for texto in tqdm.tqdm(textos_gerados):
    explicacao = explainer.explain_instance(
        text_instance=texto,
        classifier_fn=predict_prob,
        num_features=5
    )
    tokens_e_pesos = explicacao.as_list()

    tokens = [token for token, _ in tokens_e_pesos]
    pesos = [round(score, 4) for _, score in tokens_e_pesos]

    tabela_resultado.append({
        "Texto gerado": texto,
        "Tokens mais influentes": ", ".join(tokens),
        "Contribuição (LIME)": ", ".join(map(str, pesos))
    })

# Converte para DataFrame
df_resultado = pd.DataFrame(tabela_resultado)


df_resultado.to_csv("resultado_lime.csv", sep="\t")