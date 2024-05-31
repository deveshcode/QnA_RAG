import streamlit as st
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from opensearchpy import OpenSearch
from openai import OpenAI
import boto3

index_catalog = st.secrets["index_catalog"]

openai_client = OpenAI(api_key=st.secrets["openai"])
# Define the Titan model and inference parameters

# Create a session with the AWS credentials
session = boto3.Session(
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"],
    region_name='us-east-1'  # Replace with your desired region
)

# Define the Titan model and inference parameters
bedrock = session.client(service_name='bedrock-runtime')
titan_model_id = 'amazon.titan-text-lite-v1'
titan_inference_params = {
    'maxTokenCount': 100,
    'stopSequences': [],
    'temperature': 0.5,
    'topP': 0.9
}

def get_answer_openai(question, context, prompt_init):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{prompt_init}\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"}
        ]
    )
    return response.choices[0].message.content

def get_answer_titan(question, context, prompt_init):
    body = json.dumps({'inputText': f"{prompt_init}\n\nContext: {context}\n\nQuestion: {question}\nAnswer:", 'textGenerationConfig': titan_inference_params})
    response = bedrock.invoke_model(modelId=titan_model_id, body=body)
    response_body = json.loads(response.get('body').read())
    return response_body.get('results')[0].get('outputText')

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
    return [(hit['_source']['question'], hit['_source']['answer']) for hit in response['hits']['hits']]

def perform_knn_search_catalog(os_client, query, model, index_name=index_catalog, top_n=5):
    query_vector = model.encode(query).tolist()
    knn_query = {
        "size": 5,
        "query": {
            "knn": {
                "description_embedding": {
                    "vector": query_vector,
                    "k": 5
                }
            }
        }
    }
    response = os_client.search(index=index_name, body=knn_query) # change the index name according to selected option
    return [(hit['_source']['course'], hit['_source']['description']) for hit in response['hits']['hits']]
