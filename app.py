import streamlit as st
import pandas as pd
import numpy as np
from openai.embeddings_utils import cosine_similarity
import openai
import ast
import matplotlib.pyplot as plt

openai.api_key =  st.secrets["mykey"]

# Load the dataset with pre-calculated question embeddings
df = pd.read_csv("qa_dataset_with_embeddings.csv")

# Convert the string embeddings back to lists
df['Question_Embedding'] = df['Question_Embedding'].apply(ast.literal_eval)

# Function to get embedding for a new question
def get_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

# Function to find the best answer
def find_best_answer(user_question):
    # Get embedding for the user's question
    user_question_embedding = get_embedding(user_question)

    # Calculate cosine similarities for all questions in the dataset
    df['Similarity'] = df['Question_Embedding'].apply(lambda x: cosine_similarity(x, user_question_embedding))

    # Find the most similar question and get its corresponding answer
    most_similar_index = df['Similarity'].idxmax()
    max_similarity = df['Similarity'].max()

    # Set a similarity threshold to determine if a question is relevant enough
    similarity_threshold = 0.6  # You can adjust this value

    if max_similarity >= similarity_threshold:
        best_answer = df.loc[most_similar_index, 'Answer']
        return best_answer, max_similarity
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?", max_similarity

# Streamlit interface
st.title("Heart, Lung, and Blood Health Q&A")

# Input field for user's question
user_question = st.text_input("Ask a question about heart, lung, or blood health:")

# Button to trigger the answer search
if st.button("Get Answer"):
    if user_question:
        answer, similarity = find_best_answer(user_question)
        st.write(f"**Answer:** {answer}")
        st.write(f"**Similarity Score:** {similarity:.2f}")
    else:
        st.write("Please enter a question.")

# Optional additional features
if st.button("Clear"):
    st.text_input("Ask a question about heart, lung, or blood health:", value="", key="user_question")

# You can add more features like FAQs or a search bar if desired
