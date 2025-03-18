
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import hdbscan
import numpy as np
import matplotlib.pyplot as plt

# 1. Gerar embeddings com LaBSE
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/LaBSE')
sentencas = [
    "O gato está dormindo no sofá.",
    "O cachorro está brincando com a bola.",
    "O céu está azul hoje",
    "A previsão é de chuva amanhã",
    # Adicione suas sentenças aqui
]
embeddings = model.encode(sentencas)

# 2. Redução de dimensionalidade (PCA + UMAP)
pca = umap.UMAP(n_components=50, n_neighbors=2, metric='cosine', random_state=42)
embeddings_pca = pca.fit_transform(embeddings)

reducer = umap.UMAP(n_components=5, n_neighbors=2, metric='cosine', random_state=42)
embeddings_umap = reducer.fit_transform(embeddings_pca)
# Aqui demonstrado o método do cotovelo:
inertia = []
K_range = range(2, 5)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km_clusters = km.fit(embeddings_pca)
    inertia.append(km.inertia_)
plt.plot(K_range, inertia)