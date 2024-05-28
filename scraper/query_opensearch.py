from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch
import json
import os

def perform_knn_search(os_client, index_name, query, model):
    query_vector = model.encode(query).tolist()  # Ensure query vector matches the dimensionality of the vectors in the index

    knn_query = {
        "size": 5,
        "query": {
            "knn": {
                "question_embedding": {
                    "vector": query_vector,
                    "k": 5
                }
            }
        }
    }
    response = os_client.search(index=index_name, body=knn_query)
    print("\nSearch Response:")
    for hit in response['hits']['hits']:
        print(f"Question: {hit['_source']['question']}, Answer: {hit['_source']['answer']}, Score: {hit['_score']}")

# Initialize OpenSearch client
os_client = OpenSearch(
    hosts=['https://search-faq-chatbot-jzwpe6i7iz5elujpadeanj6fby.us-east-2.es.amazonaws.com'],
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

# Load pre-trained sentence transformer model for querying
model = SentenceTransformer('all-MiniLM-L6-v2')

# Perform k-NN search
perform_knn_search(os_client, "faqs_v3", "What is Data Bricks ?", model)
