import streamlit as st
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from opensearchpy import OpenSearch
from openai import OpenAI
import os

# oai_key = os.getenv('OPENAI_API_KEY')
# client = OpenAI(api_key=oai_key)
client = OpenAI(api_key=st.secrets["openai"])

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def get_answer(question, context):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"}
        ]
    )
    return response.choices[0].message.content

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
    return [(hit['_source']['question'], hit['_source']['answer']) for hit in response['hits']['hits']]

def main():
    st.set_page_config(page_title="QnA Chatbot", page_icon="💬", layout="wide")
    st.title("💬 QnA Chatbot")
    st.write("This chatbot answers FAQs scraped from selected websites.")
    
    st.sidebar.title("Navigation")
    st.sidebar.write("Use this sidebar to navigate through the app.")
    st.sidebar.header("Select a Website")
    
    data = load_data('vectorized_faqs.json')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index_name = "faqs_v3"
    
    os_client = OpenSearch(
        hosts=['https://search-faq-chatbot-jzwpe6i7iz5elujpadeanj6fby.us-east-2.es.amazonaws.com'],
        http_auth=(st.secrets["esuser"], st.secrets["espass"]       )
    )

    websites = list(data.keys())
    selected_website = st.sidebar.selectbox("Choose a website", websites)

    if selected_website:
        faqs = data[selected_website]

        st.header("Ask a Question")
        question = st.text_input("Ask a question based on the selected website's FAQs")

        if question:
            st.subheader("Intermediate Steps")
            st.write("**Initial Question:**")
            st.write(question)

            st.subheader("Results without Context")
            answer_without_context = get_answer(question, context="")
            st.write(answer_without_context)

            similar_faqs = perform_knn_search(os_client, index_name, question, model)
            context = "\n".join([f"Q: {q}\nA: {a}" for q, a in similar_faqs])

            st.write("**Appended Context:**")
            st.code(context, language='plaintext')

            st.subheader("Results with Context")
            answer_with_context = get_answer(question, context)
            st.write(answer_with_context)

            st.subheader("Comparison")
            st.write("**Answer without Context:**")
            st.write(answer_without_context)
            st.write("**Answer with Context:**")
            st.write(answer_with_context)

if __name__ == "__main__":
    main()
