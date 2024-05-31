import streamlit as st
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from opensearchpy import OpenSearch
from openai import OpenAI
import boto3
import helper

os_client = OpenSearch(
        hosts=[st.secrets["esurl"]],
        http_auth=(st.secrets["esuser"], st.secrets["espass"])
    )
aws_access_key_id = st.secrets["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws_secret_access_key"]

# Create a session with the AWS credentials
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name='us-east-1'  # Replace with your desired region
)

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
    # data = load_data('vectorized_faqs.json')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index_name = "faqs_v3"
    websites = ["databricks", "course_catalog"]
    selected_website = st.sidebar.selectbox("Choose a website", websites)


    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if selected_website:
        prompt = st.chat_input("Ask a question based on the selected website's FAQs")
        if prompt:
            with st.chat_message("user"):
                st.markdown(f"**User:** {prompt}")
            st.session_state.messages.append({"role": "user", "content": f"**User:** {prompt}"})

            # Perform KNN search and display intermediate results

            if selected_website == "databricks":
                similar_faqs = helper.perform_knn_search(os_client, index_name, prompt, model)
                prompt_init = "Answer the question based on the context"

            elif selected_website == "course_catalog":
                similar_faqs = helper.perform_knn_search_catalog(os_client, prompt, model)
                prompt_init = "Answer the question based on the context and also give the course description"

            context = "\n\n".join([f"**Q:** {q}\n**A:** {a}" for q, a in similar_faqs])

            if debug_mode:
                intermediate_message = f"**Top 5 Relevant Chunks:**\n\n{context}"
                with st.chat_message("assistant"):
                    st.markdown(intermediate_message)
                st.session_state.messages.append({"role": "assistant", "content": intermediate_message})

                # Show the appended prompt being sent to the API
                appended_prompt = f"{prompt_init}\n\nContext: {context}\n\nQuestion: {prompt}\nAnswer:"
                with st.chat_message("assistant"):
                    st.markdown(f"**Appended Prompt Sent to API:**\n\n```\n{appended_prompt}\n```")
                st.session_state.messages.append({"role": "assistant", "content": f"**Appended Prompt Sent to API:**\n\n```\n{appended_prompt}\n```"})

            if llm_option == "OpenAI GPT-3.5-turbo":
                answer = helper.get_answer_openai(prompt, context, prompt_init)
            else:
                answer = helper.get_answer_titan(prompt, context, prompt_init)
            
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant:** {answer}")
            st.session_state.messages.append({"role": "assistant", "content": f"**Assistant:** {answer}"})


if __name__ == "__main__":
    main()
        