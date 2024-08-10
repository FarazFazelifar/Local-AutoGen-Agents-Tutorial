import os
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import faiss
from PyPDF2 import PdfReader
from tqdm import tqdm
import pickle

class RAGAgent:
    def __init__(self, retriever_model: str, generator_model: str, chunk_size: int = 512, overlap: int = 50):
        print("Initializing RAGAgent...")
        self.retriever_model = retriever_model
        self.generator_model = generator_model
        self.chunk_size = chunk_size
        self.overlap = overlap

        print("Loading retriever model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(retriever_model)
        self.retriever = AutoModel.from_pretrained(retriever_model)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.retriever.to(self.device)

        print("Initializing generator...")
        self.generator = pipeline("text-generation", model=generator_model, device=self.device)

        self.index = None
        self.document_chunks = []
        self.metadata = []

    def chunk_text(self, text: str):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk = tokens[i:i + self.chunk_size]
            chunks.append(self.tokenizer.decode(chunk))
            if i + self.chunk_size >= len(tokens):
                break
        return chunks

    def build_faiss_index(self, documents: list):
        print("Building FAISS index...")
        self.document_chunks = []
        self.metadata = []
        embeddings = []

        for doc_text, meta in tqdm(documents, desc="Processing documents"):
            chunks = self.chunk_text(doc_text)
            for chunk in chunks:
                self.document_chunks.append(chunk)
                self.metadata.append(meta)
                inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=self.chunk_size).to(self.device)
                with torch.no_grad():
                    outputs = self.retriever(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(embedding)

        if not embeddings:
            raise ValueError("No embeddings were created. Check if the documents list is empty or if the chunking process is working correctly.")

        embeddings = np.array(embeddings).astype('float32')
        print(f"Shape of embeddings: {embeddings.shape}")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        print("FAISS index built successfully.")

    def retrieve(self, query: str, top_k: int = 5):
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=self.chunk_size).to(self.device)
        with torch.no_grad():
            outputs = self.retriever(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype('float32')

        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        return [(self.document_chunks[i], self.metadata[i]) for i in indices[0]]

    def generate(self, context: str, max_new_tokens: int = 150):
        response = self.generator(context, max_new_tokens=max_new_tokens, num_return_sequences=1, do_sample=True)[0]['generated_text']
        return response.strip()

    def query(self, query: str, top_k: int = 5, max_new_tokens: int = 150):
        print(f"Processing query: {query}")
        top_chunks_with_meta = self.retrieve(query, top_k)
        context = query + " " + " ".join([chunk for chunk, meta in top_chunks_with_meta])
        response = self.generate(context, max_new_tokens)
        
        citations = "\nCited from:\n" + "\n".join([f"{meta['file_name']} (Page {meta['page']})" for chunk, meta in top_chunks_with_meta])
        return response + citations

    def load_documents(self, data_dir: str):
        print(f"Loading documents from {data_dir}...")
        documents = []
        for filename in tqdm(os.listdir(data_dir), desc="Processing files"):
            if filename.endswith(".pdf"):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'rb') as file:
                        pdf_reader = PdfReader(file)
                        for page_num, page in enumerate(pdf_reader.pages, 1):
                            text = page.extract_text()
                            if text.strip():
                                documents.append((text, {'file_name': filename, 'page': page_num}))
                            else:
                                print(f"Warning: Page {page_num} in {filename} is empty.")
                except Exception as e:
                    print(f"Error reading file {filename}: {str(e)}")
            elif filename.endswith(".txt"):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                        if text.strip():
                            documents.append((text, {'file_name': filename, 'page': 1}))
                        else:
                            print(f"Warning: File {filename} is empty.")
                except Exception as e:
                    print(f"Error reading file {filename}: {str(e)}")
        
        if not documents:
            raise ValueError(f"No valid documents found in the directory: {data_dir}")
        
        print(f"Number of documents loaded: {len(documents)}")
        self.build_faiss_index(documents)
        print("Document loading and indexing complete.")

    def save(self, filepath: str):
        """Save the RAGAgent state to a file."""
        state = {
            'document_chunks': self.document_chunks,
            'metadata': self.metadata,
            'index': faiss.serialize_index(self.index) if self.index else None
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"RAGAgent state saved to {filepath}")

    def load(self, filepath: str):
        """Load the RAGAgent state from a file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.document_chunks = state['document_chunks']
        self.metadata = state['metadata']
        if state['index']:
            self.index = faiss.deserialize_index(state['index'])
        print(f"RAGAgent state loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    print("Starting RAGAgent example...")
    rag_agent = RAGAgent("sentence-transformers/all-MiniLM-L6-v2", "gpt2")

    # Check if a saved state exists
    save_path = "rag_agent_state.pkl"
    if os.path.exists(save_path):
        rag_agent.load(save_path)
    else:
        rag_agent.load_documents("path/to/your/documents")
        rag_agent.save(save_path)

    query = "What is the capital of France?"
    print(f"Querying: {query}")
    response = rag_agent.query(query)
    print("Response:")
    print(response)