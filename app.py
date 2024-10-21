import pathway as pw
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import os
import faiss  # Make sure you have FAISS installed
from secret_key import API_tog as API_KEY
import torch
import requests





tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Function to chunk document
def chunk_document(doc, chunk_size=512, overlap_size=128):
    tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=chunk_size * 10)  # Tokenize document
    chunks = []
    for i in range(0, len(tokens.input_ids[0]), chunk_size - overlap_size):
        chunk = tokens.input_ids[:, i:i + chunk_size] 
        if chunk.size(1) > 0:
            chunks.append(tokenizer.decode(chunk[0], skip_special_tokens=True))  # Decode the chunk to text
    return chunks

# Function to encode text chunks into embeddings
def encode_text_chunk(text_chunk):
    with torch.no_grad():
        outputs = model(input_ids=text_chunk)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to create a FAISS index from chunk embeddings
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]  # Get the dimension of the embeddings
    index = faiss.IndexFlatL2(dim)  # Create a FAISS index for L2 distance
    index.add(embeddings)  # Add embeddings to the index
    return index

# Function to upload and read documents
def upload_documents():
    documents = []
    print("Enter the paths of the documents you want to upload (type 'done' when finished):")
    while True:
        file_path = input("Document path: ")
        if file_path.lower() == 'done':
            break
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    return documents



def get_query():
    query = input("Enter your query: ")
    return query



def relevent_chunks(retrieved_chunks):
    combined_text = " ".join(retrieved_chunks)  # Combine the retrieved chunks into a single response
    return f"The top relevant chunks: {combined_text}"


def generate_response(retrieved_chunks,query):

    input_text = retrieved_chunks

    # Create the prompt for the API request
    prompt = f"Context information:\n{input_text}\n\nBased on the above context, please provide a detailed and accurate explanation for the following question:\n\nQuestion: {query + '?'}\n\nAnswer in a well-structured and informative manner:"

    # Prepare the API request
    headers = {
        "Authorization": f"Bearer {os.environ.get('API_KEY', 'default')}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": prompt,
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1", 
        "max_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "stop": ["</s>"],
        "stream": False
    }

    # Send the POST request to TogetherAI's API endpoint
    response = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=data)

    #print(f"Status Code: {response.status_code}")
    #print(f"Response Content: {response.text}")

    # Check for any errors in the response
    if response.status_code == 200:
        try:
            generated_text = response.json()['choices'][0]['text']
            return generated_text
        except KeyError as e:
            print(f"Error parsing the response: {e}")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

if __name__ == '__main__':
    os.environ["TOGETHER_API_KEY"] = API_KEY

    documents = upload_documents()

    
    chunked_documents = [chunk_document(doc) for doc in documents]
    all_chunks = [chunk for doc_chunks in chunked_documents for chunk in doc_chunks]  # Flatten chunk list

    chunk_embeddings = []
    
    for chunk in all_chunks:
        encoded_chunk = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        chunk_embedding = encode_text_chunk(encoded_chunk.input_ids)
        chunk_embeddings.append(chunk_embedding)

    chunk_embeddings = np.vstack(chunk_embeddings)  # Convert list of arrays to a 2D array
    faiss_index = create_faiss_index(chunk_embeddings)
    query = get_query()

    query_encoded = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
    query_embedding = encode_text_chunk(query_encoded.input_ids)

    k = 5  # Number of top chunks to retrieve
    distances, indices = faiss_index.search(query_embedding, k)


    retrieved_chunks = [all_chunks[i] for i in indices[0]]
    response = generate_response(retrieved_chunks,query)
    print(response)


    # Step 1: Upload documents
        # Step 2: Preprocess documents and chunk them with overlap
    # Step 3: Create embeddings for each chunk
    # Step 4: Create a FAISS index from chunk embeddings
    # Step 5: Query Input
    # Step 6: Retrieve relevant chunks using FAISS
    # Step 7: Get the top-k relevant chunks based on indices
    # Step 8: Generate response using the retrieved overlapping chunks
    # Step 9: Display results