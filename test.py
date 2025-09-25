import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Cargar el modelo de embeddings
model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"
embedder = SentenceTransformer(model_name)

# Cargar el índice FAISS y los chunks de texto
INDEX_PATH = "c:/Users/TZ002NB/Desktop/rag-fine-tunning/faiss_index.bin"
CHUNKS_PATH = "c:/Users/TZ002NB/Desktop/rag-fine-tunning/chunks.pkl"

index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    all_chunks = pickle.load(f)

def buscar_respuesta(pregunta, k=3):
    pregunta_vec = embedder.encode([pregunta], convert_to_numpy=True)
    D, I = index.search(pregunta_vec, k)
    resultados = [all_chunks[i] for i in I[0]]
    return resultados

def main():
    print("Chatbot RAG - Pregunta lo que quieras sobre tus PDFs (escribe 'salir' para terminar)")
    while True:
        pregunta = input("\nTu pregunta: ")
        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("¡Hasta luego!")
            break
        respuestas = buscar_respuesta(pregunta)
        print("\nRespuestas relevantes:")
        for idx, resp in enumerate(respuestas):
            print(f"\n[{idx+1}] {resp.strip()[:500]}")  # Muestra hasta 500 caracteres

if __name__ == "__main__":
    main()
