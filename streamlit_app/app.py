import streamlit as st
import json
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

oai_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=oai_key)

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def load_cache(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_answer(question, context):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"}
        ]
    )
    return response.choices[0].message.content

def find_similar_questions(question, faq_embeddings, model, top_n=5):
    question_embedding = model.encode(question).reshape(1, -1)
    faq_questions = list(faq_embeddings.keys())
    embeddings = np.array(list(faq_embeddings.values()))
    
    similarities = cosine_similarity(question_embedding, embeddings).flatten()
    similar_indices = similarities.argsort()[-top_n:][::-1]
    
    similar_faqs = [(faq_questions[i], similarities[i]) for i in similar_indices]
    return similar_faqs

def main():
    st.title("QnA Chatbot")
    st.write("This chatbot answers FAQs scraped from selected websites.")
    st.sidebar.title("Navigation")
    st.sidebar.write("Use this sidebar to navigate through the app.")

    st.sidebar.header("Select a Website")
    data = load_data('vectorized_faqs.json')
    faq_embeddings = load_cache('embeddings_cache.pkl')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
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

            similar_faqs = find_similar_questions(question, faq_embeddings, model)
            context_list = []
            for q, _ in similar_faqs:
                for faq in faqs:
                    if faq['question'] == q:
                        context_list.append(f"Q: {faq['question']}\nA: {faq['answer']}")
                        break
            context = "\n".join(context_list)

            st.write("**Appended Prompt:**")
            appended_prompt = f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
            st.code(appended_prompt, language='plaintext')

            answer = get_answer(question, context)
            st.subheader("Answer")
            st.write(answer)

if __name__ == "__main__":
    main()
