import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import tqdm


model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
embedding_dimension = 1024  # depends on the model


# Function to read and chunk files
def read_and_chunk_file(file_path, chunk_size=2000):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks


def load_data(new_files=None, vectordb="vectorstore"):
    embeddings = []
    file_info = []

    if new_files:
        # only process the new files and append to the database
        try:
            with open(os.path.join(vectordb, 'embeddings.pkl'), 'rb') as f:
                embeddings = pickle.load(f)
            with open(os.path.join(vectordb, 'file_info.pkl'), 'rb') as f:
                file_info = pickle.load(f)
        except FileNotFoundError:
            print("Embedding and file_info files are missing.")

    return embeddings, file_info


def load_or_create_index(index_file, existing_embeddings, embedding_dimension, create_new):
    if create_new:
        index = faiss.IndexFlatL2(embedding_dimension)
        return index
    try:
        # Load the existing index if it exists
        index = faiss.read_index(index_file)
    except:
        # Create a new index if it doesn't exist
        index = faiss.IndexFlatL2(embedding_dimension)
        if existing_embeddings:
            # Rebuild index from existing embeddings if there are any
            index.add(np.vstack(existing_embeddings))  # Assuming embeddings are stored in a list of numpy arrays
    return index


def update_embeddings(folder="test", vectordb = "vectorstore", new_files=[]):

    if not os.path.exists(vectordb):
        os.makedirs(vectordb)
    
    embeddings, file_info = load_data(new_files, vectordb)  # if process all files, will return empty list and rewrite

    index_file = os.path.join(vectordb, "faiss_index")
    create_new = True if len(new_files)==0 else False
    index = load_or_create_index(index_file, embeddings, embedding_dimension, create_new)
    
    # get the files to process, either all files or new files
    if not new_files:
        files_to_process = [f for f in os.listdir(folder) if f.endswith('.txt')]
        files_to_process = [os.path.join(folder, file) for file in files_to_process]
    else:
        files_to_process = new_files if isinstance(new_files, list) else [new_files]

    # loop through
    for filepath in tqdm.tqdm(files_to_process):
        if filepath.endswith(".txt"):
            chunks = read_and_chunk_file(filepath)
            for i, chunk in enumerate(chunks):
                embedding = model.encode(chunk, convert_to_tensor=True)
                embedding_np = embedding.cpu().numpy()
                
                # Reshape the embedding into a 2D array (1 row, N columns where N is the embedding size)
                embedding_np_reshaped = embedding_np.reshape(1, -1)   # shape (1, N)
                # print("embedding_np_reshaped.shape", embedding_np_reshaped.shape)
                
                embeddings.append(embedding_np)
                file_info.append((os.path.basename(filepath), i, chunk))
                index.add(embedding_np_reshaped)  # Add to Faiss index after reshaping

    # Save embeddings and file info
    with open(os.path.join(vectordb, 'embeddings.pkl'), 'wb') as f:
        pickle.dump(embeddings, f)
    with open(os.path.join(vectordb, 'file_info.pkl'), 'wb') as f:
        pickle.dump(file_info, f)
    faiss.write_index(index, os.path.join(vectordb, 'faiss_index'))

    if not new_files:
        print("Saved embeddings for {} chunks.".format(len(embeddings)))
    else:
        print("Added embeddings for {} files. Now total {} chunks".format(len(files_to_process), len(embeddings)))

