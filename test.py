import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
BASE_DIR = "C:/Users/Usuario/Desktop/cursos_repos/rag-finetunning"
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.pkl")

# Embeddings
MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"

# Cliente OpenAI (usa tu API key en variable de entorno OPENAI_API_KEY)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("No se encontró la variable OPENAI_API_KEY en el archivo .env")
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------
# CARGAR MODELO Y DATOS
# -------------------------------
print("Cargando modelo de embeddings...")
embedder = SentenceTransformer(MODEL_NAME, device="cpu")

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"No se encontró el índice en {INDEX_PATH}")
index = faiss.read_index(INDEX_PATH)

if not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError(f"No se encontraron los chunks en {CHUNKS_PATH}")
with open(CHUNKS_PATH, "rb") as f:
    all_chunks = pickle.load(f)

# -------------------------------
# FUNCIÓN DE BÚSQUEDA
# -------------------------------
def buscar_respuesta(pregunta, k=3):
    pregunta_vec = embedder.encode([pregunta], convert_to_numpy=True)
    D, I = index.search(pregunta_vec, k)
    resultados = [all_chunks[i] for i in I[0]]
    return resultados

# -------------------------------
# GPT CON RAG
# -------------------------------
def generar_respuesta(pregunta, contexto):
    prompt = f"""
    Responde en español a la siguiente pregunta basándote en el contexto dado.
    Si el contexto no tiene la información suficiente, indícalo claramente.

    Pregunta: {pregunta}

    Contexto:
    {contexto}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # puedes cambiar por gpt-4.1, gpt-3.5, etc.
        messages=[
            {"role": "system", "content": "Eres un asistente útil que responde SIEMPRE en español."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# -------------------------------
# LOOP PRINCIPAL
# -------------------------------
def main():
    print("Chatbot RAG - Pregunta lo que quieras sobre tus PDFs (escribe 'salir' para terminar)")
    while True:
        try:
            pregunta = input("\nTu pregunta: ")
        except KeyboardInterrupt:
            print("\nInterrumpido. ¡Hasta luego!")
            break

        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("¡Hasta luego!")
            break

        # Buscar chunks
        respuestas = buscar_respuesta(pregunta)
        contexto = "\n\n".join(respuestas)

        # Pasar a GPT
        respuesta_final = generar_respuesta(pregunta, contexto)
        print("\n📝 Respuesta del asistente:")
        print(respuesta_final)

if __name__ == "__main__":
    main()
