import streamlit as st
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from opensearchpy import OpenSearch
from openai import OpenAI
import boto3

# Set up OpenAI and Bedrock API clients
openai_client = OpenAI(api_key=st.secrets["openai"])
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

# Define the Titan model and inference parameters
titan_model_id = 'amazon.titan-text-lite-v1'
titan_inference_params = {
    'maxTokenCount': 100,
    'stopSequences': [],
    'temperature': 0.5,
    'topP': 0.9
}

# Load data function
def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# OpenAI model answer function
def get_answer_openai(question, context):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"}
        ]
    )
    return response.choices[0].message.content

# Titan model answer function
def get_answer_titan(question, context):
    body = json.dumps({'inputText': f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:", 'textGenerationConfig': titan_inference_params})
    response = bedrock.invoke_model(modelId=titan_model_id, body=body)
    response_body = json.loads(response.get('body').read())
    return response_body.get('results')[0].get('outputText')

# Perform KNN search function
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

def main():
    st.set_page_config(page_title="QnA Chatbot", page_icon="💬", layout="wide")
    st.title("💬 QnA Chatbot")
    st.write("This chatbot answers FAQs scraped from selected websites.")
    
    st.sidebar.title("Navigation")
    st.sidebar.write("Use this sidebar to navigate through the app.")
    st.sidebar.header("Select a Website")

    # LLM selection dropdown
    llm_option = st.sidebar.selectbox("Choose a Language Model", ["OpenAI GPT-3.5-turbo", "Amazon Titan"])
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    data = load_data('vectorized_faqs.json')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index_name = "faqs_v3"
    
    os_client = OpenSearch(
        hosts=['https://search-faq-chatbot-jzwpe6i7iz5elujpadeanj6fby.us-east-2.es.amazonaws.com'],
        http_auth=(st.secrets["esuser"], st.secrets["espass"])
    )

    websites = list(data.keys())
    selected_website = st.sidebar.selectbox("Choose a website", websites)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if selected_website:
        faqs = data[selected_website]

        prompt = st.chat_input("Ask a question based on the selected website's FAQs")
        if prompt:
            with st.chat_message("user"):
                st.markdown(f"**User:** {prompt}")
            st.session_state.messages.append({"role": "user", "content": f"**User:** {prompt}"})

            # Perform KNN search and display intermediate results
            similar_faqs = perform_knn_search(os_client, index_name, prompt, model)
            context = "\n\n".join([f"**Q:** {q}\n**A:** {a}" for q, a in similar_faqs])

            if debug_mode:
                intermediate_message = f"**Top 5 Relevant Chunks:**\n\n{context}"
                with st.chat_message("assistant"):
                    st.markdown(intermediate_message)
                st.session_state.messages.append({"role": "assistant", "content": intermediate_message})

                # Show the appended prompt being sent to the API
                appended_prompt = f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {prompt}\nAnswer:"
                with st.chat_message("assistant"):
                    st.markdown(f"**Appended Prompt Sent to API:**\n\n```\n{appended_prompt}\n```")
                st.session_state.messages.append({"role": "assistant", "content": f"**Appended Prompt Sent to API:**\n\n```\n{appended_prompt}\n```"})

            # Generate the final answer with context using selected LLM
            if llm_option == "OpenAI GPT-3.5-turbo":
                answer = get_answer_openai(prompt, context)
            else:
                answer = get_answer_titan(prompt, context)
            
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant:** {answer}")
            st.session_state.messages.append({"role": "assistant", "content": f"**Assistant:** {answer}"})

if __name__ == "__main__":
    main()
