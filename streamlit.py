import streamlit as st
from app import chunk_document, encode_text_chunk, create_faiss_index, generate_response
import numpy as np
import torch
import pathway as pw
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import os
import faiss  # Make sure you have FAISS installed
from secret_key import API_tog as API_KEY
import torch


# Tokenizer and Model for Embedding
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Initialize session state variables for faiss_index and all_chunks
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None

if 'all_chunks' not in st.session_state:
    st.session_state.all_chunks = []

if 'kmeans' not in st.session_state:
    st.session_state.kmeans = None

if 'chunk_embeddings' not in st.session_state:
    st.session_state.chunk_embeddings = None


# Streamlit UI
st.title("Document Query Application")

# Step 1: Upload Documents
st.header("Upload Documents")
uploaded_files = st.file_uploader("Choose text files", type="txt", accept_multiple_files=True)

documents = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        documents.append(uploaded_file.getvalue().decode("utf-8"))


# Step 2: Process Documents
if st.button("Process Documents"):
    if documents:
        # Step 3: Preprocess documents and chunk them with overlap
        chunked_documents = [chunk_document(doc) for doc in documents]
        st.session_state.all_chunks = [chunk for doc_chunks in chunked_documents for chunk in doc_chunks]  # Flatten chunk list

        # Step 4: Create embeddings for each chunk
        chunk_embeddings = []
        for chunk in st.session_state.all_chunks:
            encoded_chunk = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
            chunk_embedding = encode_text_chunk(encoded_chunk.input_ids)
            chunk_embedding = chunk_embedding.flatten()
            chunk_embeddings.append(chunk_embedding)

        st.session_state.chunk_embeddings = np.vstack(chunk_embeddings)  # Convert list of arrays to a 2D array

        num_clusters = 10  # Number of clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(chunk_embeddings)
        st.session_state.kmeans = kmeans

        # Step 5: Create a FAISS index from chunk embeddings
        #st.session_state.faiss_index = create_faiss_index(chunk_embeddings)

        # Notify user
        st.success("Documents processed and embeddings created successfully!")


# Step 6: Query Input
query = st.text_input("Enter your query:")

# Step 7: Retrieve Relevant Chunks
if st.button("Search"):
    if query:
        if st.session_state.kmeans is None or st.session_state.chunk_embeddings is None:
            st.error("process documents first")
        else:
            # Step 8: Encode the query and create its embedding
            query_encoded = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
            query_embedding = encode_text_chunk(query_encoded.input_ids).reshape(1, -1)

            cluster_centers = st.session_state.kmeans.cluster_centers_
            cluster_similarities = cosine_similarity(query_embedding, cluster_centers)
            closest_cluster = np.argmax(cluster_similarities)

            cluster_labels = st.session_state.kmeans.labels_
            cluster_indices = np.where(cluster_labels == closest_cluster)[0]

            cluster_chunk_embeddings = st.session_state.chunk_embeddings[cluster_indices]
            chunk_similarities = cosine_similarity(query_embedding, cluster_chunk_embeddings)
            

            k = 5  # Number of top chunks to retrieve
            top_k_indices_within_cluster = np.argsort(-chunk_similarities[0])[:k]
            top_k_chunk_indices = cluster_indices[top_k_indices_within_cluster]
            st.session_state.retrieved_chunks = [st.session_state.all_chunks[i] for i in top_k_chunk_indices]
            

            # Step 11: Generate response using the retrieved overlapping chunks
            response = generate_response(st.session_state.retrieved_chunks,query)

            # Step 12: Display results
            
            st.subheader("Top Relevant Chunks:")
            for i, chunk in enumerate(st.session_state.retrieved_chunks):
                st.write(f"Chunk {i + 1}: {chunk}")

            # Display the generated response
            st.subheader("Generated Response:")
            st.write(response)
    else:
        st.warning("Please enter a query.")



# how to decide if the query is unrelated to context provided by the user document make a function for that'''