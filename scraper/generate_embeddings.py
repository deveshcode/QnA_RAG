import json
import pickle
from sentence_transformers import SentenceTransformer

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def save_cache(cache, filename):
    with open(filename, 'wb') as f:
        pickle.dump(cache, f)

def load_cache(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def generate_embeddings(data, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = {}

    for website, faqs in data.items():
        for faq in faqs:
            question_embedding = model.encode(faq['question']).tolist()
            faq['question_embedding'] = question_embedding
            embeddings[faq['question']] = question_embedding
    return data, embeddings

if __name__ == "__main__":
    input_filename = 'cleaned_faqs.json'
    output_filename = 'vectorized_faqs.json'
    cache_filename = 'embeddings_cache.pkl'

    data = load_data(input_filename)
    cache = load_cache(cache_filename)

    vectorized_data, embeddings = generate_embeddings(data)
    save_data(vectorized_data, output_filename)
    save_cache(embeddings, cache_filename)

    print(f"Data converted to vectors and saved to {output_filename}")
