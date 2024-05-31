from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
import json
import os

def create_index(os_client, index_name='faqs'):
    settings = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "knn.algo_param.ef_search": 100  # Optional: Tune this parameter based on your needs
            },
            "analysis": {
                "analyzer": {
                    "default": {
                        "type": "standard"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "question": {
                    "type": "text"
                },
                "answer": {
                    "type": "text"
                },
                "question_embedding": {
                    "type": "knn_vector",
                    "dimension": 384,  # Adjust this to the dimensionality of your embeddings
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 24
                        }
                    }
                }
            }
        }
    }
    # Delete the index if it already exists
    if os_client.indices.exists(index=index_name):
        os_client.indices.delete(index=index_name)

    os_client.indices.create(index=index_name, body=settings, ignore=400)

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def index_faq_embeddings(os_client, index_name, data):
    # Prepare bulk indexing data
    actions = [
        {
            "_index": index_name,
            "_id": i,
            "_source": {
                "question": doc["question"],
                "answer": doc["answer"],
                "question_embedding": doc["question_embedding"]
            }
        }
        for i, doc in enumerate(data)
    ]

    # Bulk index data
    bulk(os_client, actions)

    # Check index mapping
    mapping_response = os_client.indices.get_mapping(index=index_name)
    print("\n\nIndex Mapping:")
    print(mapping_response)

if __name__ == "__main__":
    os_client = OpenSearch(
        hosts=['https://search-faq-chatbot-5ep7nhawvwkiqp5tow37fklyji.us-east-2.es.amazonaws.com'],
        http_auth=(os.getenv("ES_USERNAME"), os.getenv("ES_PASSWORD"))
    )

    index_name = 'faqs_v3'
    create_index(os_client, index_name)

    data = load_data('vectorized_faqs.json')["databricks"]
    index_faq_embeddings(os_client, index_name, data)

    print(f"Data indexed to {index_name}")
