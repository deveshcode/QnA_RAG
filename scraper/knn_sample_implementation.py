import json
from sentence_transformers import SentenceTransformer
import os

def generate_fruit_embeddings(fruit_names, output_file):
    # Load pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for fruit names
    data = [
        {"fruit_name": fruit_names[i], "fruit_vector": model.encode(fruit_names[i]).tolist()}
        for i in range(len(fruit_names))
    ]

    # Save embeddings to JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f)

    print(f"Embeddings for {len(fruit_names)} fruits saved to {output_file}")

# List of fruits
fruit_names = [
    "Apple", "Banana", "Cherry", "Date", "Elderberry",
    "Fig", "Grape", "Honeydew", "Indian Fig", "Jackfruit",
    "Kiwi", "Lemon", "Mango", "Nectarine", "Orange",
    "Papaya", "Quince", "Raspberry", "Strawberry", "Tomato"
]

# Generate and store embeddings
generate_fruit_embeddings(fruit_names, 'fruit_embeddings.json')

from opensearchpy import OpenSearch
import json

def load_fruit_embeddings(input_file):
    # Load embeddings from JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def index_fruit_embeddings(os_client, index_name, data):
    # Create index with k-NN vector support
    settings = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100  # Optional: Tune this parameter based on your needs
            }
        },
        "mappings": {
            "properties": {
                "fruit_vector": {
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
                },
                "fruit_name": {
                    "type": "keyword"
                }
            }
        }
    }

    # Delete the index if it already exists
    if os_client.indices.exists(index=index_name):
        os_client.indices.delete(index=index_name)

    # Create the index
    os_client.indices.create(index=index_name, body=settings)

    # Index sample data
    for i, doc in enumerate(data):
        os_client.index(index=index_name, id=i, body=doc)

    # Check index mapping
    mapping_response = os_client.indices.get_mapping(index=index_name)
    print("\n\nIndex Mapping:")
    print(mapping_response)

def perform_knn_search(os_client, index_name, query_fruit, model):
    query_vector = model.encode(query_fruit).tolist()  # Ensure query vector matches the dimensionality of the vectors in the index
    print("\n\nQuery Vector:")
    print(query_vector)

    knn_query = {
        "size": 5,
        "query": {
            "knn": {
                "fruit_vector": {
                    "vector": query_vector,
                    "k": 5
                }
            }
        }
    }

    print("\n\nk-NN Query:", knn_query)
    response = os_client.search(index=index_name, body=knn_query)
    print("\nSearch Response:")
    for hit in response['hits']['hits']:
        print(f"Fruit: {hit['_source']['fruit_name']}, Score: {hit['_score']}")

# Initialize OpenSearch client
# Initialize OpenSearch client
os_client = OpenSearch(
    hosts=['https://search-faq-chatbot-5ep7nhawvwkiqp5tow37fklyji.us-east-2.es.amazonaws.com'],
    http_auth=(os.getenv("ES_USERNAME"), os.getenv("ES_PASSWORD"))
)

# Check OpenSearch Connectivity
if not os_client.ping():
    print("Error: Failed to connect to OpenSearch.")
    exit(1)

# Check installed plugins
plugins_response = os_client.cat.plugins(format='json')
print("\n\nInstalled Plugins:")
for plugin in plugins_response:
    print(f"Name: {plugin['component']}, Version: {plugin['version']}")

index_name = 'fruits_knn_index'
data = load_fruit_embeddings('fruit_embeddings.json')

# Index the data
index_fruit_embeddings(os_client, index_name, data)

# Load pre-trained sentence transformer model for querying
model = SentenceTransformer('all-MiniLM-L6-v2')

# Perform k-NN search
perform_knn_search(os_client, index_name, "Kiwi", model)