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
BASE_DIR = "C:/Users/Usuario/Desktop/cursos_repos/Inflecta"
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.pkl")

# Embeddings
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

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
# GPT CON RAG Y CONTEXTO
# -------------------------------
def generar_respuesta(pregunta, contexto, historial):
    # Agregamos la pregunta actual con su contexto al historial
    historial.append({"role": "user", "content": f"{pregunta}\n\nContexto:\n{contexto}"})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un asistente útil que responde SIEMPRE en español y basado solo en el contexto dado."},
            *historial  # enviamos todo el historial acumulado
        ],
        temperature=0.3
    )

    respuesta = response.choices[0].message.content.strip()
    historial.append({"role": "assistant", "content": respuesta})  # guardamos la respuesta también

    # Evitar que el historial crezca demasiado
    if len(historial) > 10:
        historial = historial[-10:]

    return respuesta, historial

# -------------------------------
# LOOP PRINCIPAL
# -------------------------------
def main():
    print("Chatbot RAG - Pregunta lo que quieras sobre tus PDFs (escribe 'salir' para terminar)")

    historial = []  # memoria temporal de la conversación

    while True:
        try:
            pregunta = input("\nTu pregunta: ")
        except KeyboardInterrupt:
            print("\nInterrumpido. ¡Hasta luego!")
            break

        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("¡Hasta luego!")
            break

        # Buscar chunks relevantes
        respuestas = buscar_respuesta(pregunta)
        contexto = "\n\n".join(respuestas)

        # Generar respuesta con memoria
        respuesta_final, historial = generar_respuesta(pregunta, contexto, historial)

        print("\n📝 Respuesta del asistente:")
        print(respuesta_final)

if __name__ == "__main__":
    main()
