from pymilvus import connections, Collection
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# ==============================
# 1. Connexion à Milvus
# ==============================
connections.connect("default", host="localhost", port="19530")
print("Connecté à Milvus")

collection_name = "chatbot_chunks_hybrid"
collection = Collection(collection_name)
collection.load()

# ==============================
# 2. Charger le modèle
# ==============================
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
print("BGEM3EmbeddingFunction initialisé")

# ==============================
# 3. Fonction de recherche hybride (dense + sparse + fusion RRF)
# ==============================
def hybrid_search(query, top_k=5, alpha=0.5,return_passages=True):
    # Générer embeddings
    q_emb = ef([query])

    dense_vec = q_emb["dense"][0].tolist()

    coo = q_emb["sparse"][0].tocoo()
    sparse_vec = {int(i): float(v) for i, v in zip(coo.col, coo.data)}
    # Recherche dense
    dense_results = collection.search(
        data=[dense_vec],
        anns_field="dense",
        param={"metric_type": "IP", "params": {"ef": 64}},
        limit=top_k * 2,  # on prend plus large avant fusion
        output_fields=["source", "chunk_index", "text"],
    )[0]

    # Recherche sparse
    sparse_results = collection.search(
        data=[sparse_vec],
        anns_field="sparse",
        param={"metric_type": "IP"},
        limit=top_k * 2,
        output_fields=["source", "chunk_index", "text"],
    )[0]

    # Fusion via Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    for rank, hit in enumerate(dense_results):
        key = (hit.entity.get("source"), hit.entity.get("chunk_index"))
        rrf_scores[key] = rrf_scores.get(key, 0) + alpha / (rank + 60)

    for rank, hit in enumerate(sparse_results):
        key = (hit.entity.get("source"), hit.entity.get("chunk_index"))
        rrf_scores[key] = rrf_scores.get(key, 0) + (1 - alpha) / (rank + 60)

    # Trier par score fusionné
    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for (source, chunk_idx), score in merged[:top_k]:
        entity = next(hit.entity for hit in (dense_results + sparse_results)
                      if hit.entity.get("source") == source and hit.entity.get("chunk_index") == chunk_idx)
        results.append({
            "source": source,
            "chunk_index": chunk_idx,
            "text": entity.get("text"),
            "score": score
        })

    if return_passages:
        return results

    # sinon, affichage classique
    print("\nRésultats fusionnés (RRF) pour :", query)
    for i, r in enumerate(results, 1):
        print(f"{i}. [Score={r['score']:.4f}] {r['source']} (chunk {r['chunk_index']})")
        print(f"   → {r['text']}...\n")
# ==============================
# 4. Questions de test
# ==============================
# questions = [
#     "Quels sont les droits du travailleur malade ?",
#     "Quelles sont les conditions de licenciement ?",
#     "Que se passe-t-il en cas de décès de l’agent ?",
#     "Comment est calculée l’indemnité de congé ?",
#     "Quels sont les motifs de rupture du contrat de travail ?",
# ]

# for q in questions:
#     hybrid_search(q, top_k=5, alpha=0.5)
