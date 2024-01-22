import streamlit as st
import pandas as pd 
from sentence_transformers import SentenceTransformer, util
import numpy as np
model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv('sample.csv')

def compare_sentences(sentence_1=str, sentence_2=str, model_name=str, embedding_type="cls_token_embedding", metric="cosine") -> str:
    """Utilizes an NLP model that calculates the similarity between 
    two sentences or phrases."""
    #Compute embedding for both lists
    embeddings1 = model.encode(sentence_1, convert_to_tensor=True)
    embeddings2 = model.encode(sentence_2, convert_to_tensor=True)

    # #Compute cosine-similarities
    cosine_scores= util.cos_sim(embeddings1, embeddings2)

    # #Output the pairs with their score
    return    cosine_scores[0][0]

def compare_scores():
    similarity_scores =[]
    max_index = 0
    similarity_scores = df['Question'].apply(compare_sentences, args=(input, model) )
    max_index = np.argmax(similarity_scores)
    st.write(df.loc[max_index,['Syntax']])

model_1 = "sentence-transformers/all-MiniLM-L6-v2"
st.title("Enter a Question")
input = st.text_input("Search for a query")
if input:
    st.write('Entered query is -->', input)
    compare_scores()
else:
    st.write('No query entered')
compare_scores()


