import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

with open('output/embeddings.json', 'r', encoding='utf-8') as f:
    embeddings_data = json.load(f)

embeddings = np.array(embeddings_data['embeddings'])
indices = embeddings_data['indices']
pages = embeddings_data['pages']

model = SentenceTransformer('thenlper/gte-small')

def search_query(query):
    query_embedding = model.encode(query, convert_to_tensor=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_embedding = query_embedding.to(device)

    embeddings_tensor = torch.tensor(embeddings, dtype=query_embedding.dtype).to(device)

    similarities = util.pytorch_cos_sim(query_embedding, embeddings_tensor)[0]

    most_similar_index = np.argmax(similarities.cpu().numpy())

    page_number = pages[most_similar_index]
    chunk_index = indices[most_similar_index]

    return page_number, chunk_index

query = "Il figlio naturale succede all'ascendente legittimo immediato del suo genitore"
page, index = search_query(query)
print(f"Query found in page: {page}, chunk index: {index}")