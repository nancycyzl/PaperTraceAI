import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import ollama
import argparse
from openai import OpenAI
import gradio as gr

# Load the model for encoding the query
model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)


# Function to load embeddings, file_info, and index
def load_data(vectordb):
    with open(os.path.join(vectordb, 'embeddings.pkl'), 'rb') as f:
        embeddings = pickle.load(f)
    with open(os.path.join(vectordb, 'file_info.pkl'), 'rb') as f:
        file_info = pickle.load(f)
    index = faiss.read_index(os.path.join(vectordb,'faiss_index'))
    return embeddings, file_info, index


def distance_to_similarity(distances):
    # Convert L2 distance to cosine similarity
    return 1 - (distances / 2)


def find_relevant_chunks(question, index, file_info, similarity_threshold=0.8, min_k=10):
    question_embedding = model.encode(question, convert_to_tensor=True)
    question_embedding_np = question_embedding.cpu().numpy().reshape(1, -1)  # Ensure correct shape

    # Calculate a rough threshold for L2 distance that would correspond to the desired cosine similarity
    l2_threshold = 2 * (1 - similarity_threshold)
    
    # Performing range search using the L2 threshold
    lims, D, I = index.range_search(question_embedding_np, l2_threshold)
    
    # Initialize results
    results = []

    # Iterating through the indices and distances returned by range_search
    for i in range(len(lims) - 1):
        start_idx = lims[i]
        end_idx = lims[i+1]
        for j in range(start_idx, end_idx):
            dist = D[j]
            idx = I[j]
            cosine_similarity = 1 - (dist / 2)  # Converting distance to cosine similarity
            if cosine_similarity >= similarity_threshold:
                filename, chunk_idx, chunk_text = file_info[idx]
                results.append((filename, chunk_idx, cosine_similarity, chunk_text))

    # Sort results by similarity in descending order
    results.sort(key=lambda x: x[2], reverse=True)

    # If fewer than min_k results, return the top min_k chunks
    if len(results) < min_k:
        # Perform a k-nearest neighbor search to fill up the remaining slots
        D, I = index.search(question_embedding_np, k=min_k)
        for score, index in zip(D[0], I[0]):
            sim_score = 1 - (score / 2)
            if len(results) < min_k:
                filename, chunk_idx, chunk_text = file_info[index]
                if (filename, chunk_idx) not in [x[:2] for x in results]:  # Ensure not already included
                    results.append((filename, chunk_idx, sim_score, chunk_text))
            else:
                break

    return results


# Example usage
def get_relevant_chunks(question, similarity_threshold, min_k):
    embeddings, file_info, index = load_data(vectordb = "vectorstore")
    results = find_relevant_chunks(question, index, file_info, similarity_threshold, min_k)
    print("Total returned chunks: ", len(results))
    return results


def get_response(question, threshold, min_k, model_name, verbose):
    context = "Here are some sources and their content.\n"
    chunks = get_relevant_chunks(question, threshold, min_k)
    for filename, _, _, chunk_text in chunks:
        context += "Source: {}\nContent: {}\n\n".format(filename, chunk_text)

    # LLM
    system_prompt = ""
    user_prompt = context + '''\n\n
                            Some of context provided may not be relevant to the question. Only include those related to the question.
                            Include the source (txt files) as the reference.
                            Question: {} \n\n
                            '''.format(question)
    if "llama" in model_name.lower():
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                "role": "system",
                "content": system_prompt,
                },
                {
                "role": "user",
                "content": user_prompt,
                }
            ])

        response_string = response['message']['content']

    elif "gpt" in model_name.lower():
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ])
        response_string = response.choices[0].message.content

    else:
        raise ValueError(f"Model {model_name} is not supported!")

    if verbose:
        additional_string = "Chunks returned:\n"
        additional_string += context
        additional_string += "\n\n"
        additional_string += "-------------------" * 5
        additional_string += "\n\n LLM output:"
        response_string = additional_string + response_string

    return response_string


def gradio_interface(question, threshold, min_k, model_name, verbose):
    response = get_response(question, threshold, min_k, model_name, verbose)
    return response


# Create Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Question"),
        gr.Slider(0.5, 1.0, value=0.7, label="Similarity Threshold"),
        gr.Slider(1, 20, value=10, step=1, label="Minimum Chunks to return"),
        gr.Dropdown(["llama3.1", "gpt-4o-mini"], label="Model", value="llama3.1"),
        gr.Checkbox(label="Verbose")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="PaperTraceAI",
    description="Ask questions and trace back in the database."
)

if __name__ == "__main__":
    iface.launch()


