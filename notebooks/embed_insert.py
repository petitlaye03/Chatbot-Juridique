import json
from pathlib import Path
import torch
from pymilvus import connections, Collection
from pymilvus.model.hybrid import BGEM3EmbeddingFunction 

# ==============================
# 1. CPU uniquement
# ==============================
device = "cpu"
torch.set_num_threads(6)

# ==============================
# 2. Connexion à Milvus
# ==============================
connections.connect("default", host="localhost", port="19530")
print("Connecté à Milvus")

collection_name = "chatbot_chunks_hybrid"
collection = Collection(collection_name)
collection.load()

# ==============================
# 3. Charger les JSON
# ==============================
data_dir = Path("data")
files = [data_dir / "code_travail_chunks.json", data_dir / "manuel_chunks.json"]

documents = []
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        docs = json.load(f)
        documents.extend(docs)

print(f"{len(documents)} chunks chargés")

# ==============================
# 4. Charger le modèle BGE-M3 via Milvus SDK
# ==============================
ef = BGEM3EmbeddingFunction(use_fp16=False, device=device)
print(f"BGEM3EmbeddingFunction initialisé sur {device.upper()}")

# ==============================
# 5. Générer et insérer les embeddings par batch
# ==============================
texts = [doc["text"] for doc in documents]
sources = [doc["source"] for doc in documents]
indices = [doc["chunk_index"] for doc in documents]

BATCH_SIZE = 50
print(f"Insertion par batch de {BATCH_SIZE} chunks...")

for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_sources = sources[i:i+BATCH_SIZE]
    batch_indices = indices[i:i+BATCH_SIZE]

    # Génération des embeddings pour ce batch
    embeddings = ef(batch_texts)
    dense_vectors = embeddings["dense"]
    sparse_vectors = embeddings["sparse"]

    # Préparer entités
    entities = [
        batch_sources,     
        batch_indices,     
        batch_texts,       
        dense_vectors,     
        sparse_vectors     
    ]

    # Insertion
    collection.insert(entities)
    print(f"Batch {i//BATCH_SIZE + 1} inséré ({len(batch_texts)} chunks)")

collection.flush()
print("Insertion terminée")
print(f"Nombre total d’entrées : {collection.num_entities}")
