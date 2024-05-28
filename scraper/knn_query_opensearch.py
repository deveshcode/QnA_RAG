from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
import os

def search_similar_questions(os_client, query, model, index_name='faqs', top_n=5):
    # Encode the query to get the query vector
    query_vector = model.encode(query).tolist()
    # Debug: Print the query vector to check its format
    print("Query Vector:", query_vector)

    # Ensure the query vector is correctly formatted as a list of floats
    if not isinstance(query_vector, list):
        raise ValueError("query_vector should be a list of floats")

    knn_query = {
        "size": top_n,
        "query": {
            "knn": {
                "field": "question_embedding",
                "query_vector": query_vector,
                "k": top_n,
                "num_candidates": 10
            }
        },
        "_source": ["question", "answer"]
    }

    response = os_client.search(index=index_name, body=knn_query)

    return [(hit['_source']['question'], hit['_source']['answer']) for hit in response['hits']['hits']]

if __name__ == "__main__":
    os_client = OpenSearch(
        hosts=['https://search-faq-chatbot-jzwpe6i7iz5elujpadeanj6fby.us-east-2.es.amazonaws.com'],
        http_auth=(os.getenv("ES_USERNAME"), os.getenv("ES_PASSWORD"))
    )

    model = SentenceTransformer('all-MiniLM-L6-v2')
    query = "What is Databricks?"
    similar_questions = search_similar_questions(os_client, query, model)

    for question, answer in similar_questions:
        print(f"Q: {question}\nA: {answer}\n")

# Debugging and plugin check
# curl -X GET "https://search-faq-chatbot-jzwpe6i7iz5elujpadeanj6fby.us-east-2.es.amazonaws.com/_cat/plugins?v&pretty" -u devesh:Dev@sh654321
# curl -X GET "https://search-faq-chatbot-jzwpe6i7iz5elujpadeanj6fby.us-east-2.es.amazonaws.com/faqs/_doc/1" -u devesh:Dev@sh654321
