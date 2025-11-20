from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# 1. Connexion à Milvus
connections.connect("default", host="localhost", port="19530")
print("Connecté à Milvus")

# 2. Paramètres des embeddings
DENSE_DIM = 1024 

# 3. Définir le schéma de la collection hybride
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200),   
    FieldSchema(name="chunk_index", dtype=DataType.INT64),                
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8000),    
    FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=DENSE_DIM),  
    FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),         
]

schema = CollectionSchema(fields, description="Collection hybride (dense + sparse) pour chatbot juridique")

# 4. Créer la collection 
collection_name = "chatbot_chunks_hybrid"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Ancienne collection '{collection_name}' supprimée")

collection = Collection(name=collection_name, schema=schema)
print(f"Nouvelle collection '{collection_name}' créée avec champs dense et sparse")

# 5. Créer les index
# Index dense (vecteur continu)
index_params_dense = {
    "index_type": "HNSW",   
    "metric_type": "IP",    
    "params": {"M": 8, "efConstruction": 64}
}
collection.create_index(field_name="dense", index_params=index_params_dense)
print("Index HNSW créé sur 'dense'")

# Index sparse (vecteur lexical)
index_params_sparse = {
    "index_type": "SPARSE_INVERTED_INDEX",
    "metric_type": "IP",
    "params": {}
}
collection.create_index(field_name="sparse", index_params=index_params_sparse)
print("Index inversé créé sur 'sparse'")

# 6. Charger la collection
collection.load()
print("Collection hybride prête à recevoir les données")
