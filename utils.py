from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


# Carregando o modelo e o tokenizer
model_name = "microsoft/mdeberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Função que gera embeddings a partir das sentenças
def gerar_embedding(sentenca):
    tokens = tokenizer(sentenca, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.numpy()

# Gerar embeddings para um conjunto de sentenças
def gerar_embeddings_conjunto(conjunto_sentencas):
    return np.vstack([gerar_embedding(s) for s in conjunto_sentencas])

# Agrupamento de sentenças utilizando clustering hierárquico
def agrupar_sentencas(conjunto_sentencas, n_clusters):
    embeddings = gerar_embeddings_conjunto(conjunto_sentencas)
    matriz_similaridade = cosine_similarity(embeddings)

    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
    agrupamentos = clustering.fit_predict(1 - matriz_similaridade)
    return agrupamentos