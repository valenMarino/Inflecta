
## Recomendación: Usar entorno virtual (venv)

Se recomienda crear y activar un entorno virtual para aislar las dependencias del proyecto. En Windows, ejecuta en PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Luego instala las dependencias desde el notebook o manualmente con pip.

# RAG Fine-Tuning: Procesamiento y Búsqueda Semántica en PDFs

Este proyecto permite extraer texto de archivos PDF (incluyendo escaneados mediante OCR), generar embeddings semánticos multilenguaje y realizar búsquedas inteligentes sobre el contenido. Está orientado a experimentos de Retrieval-Augmented Generation (RAG) y utiliza modelos ligeros compatibles con Ollama.

## Características principales
- Procesamiento de PDFs, incluyendo imágenes escaneadas (OCR con Tesseract).
- Vectorización de texto usando `sentence-transformers` multilenguaje.
- Indexación eficiente con FAISS para búsquedas semánticas rápidas.
- Ejemplo de consulta y respuesta tipo asistente.

## Estructura del proyecto
- `Notebook.ipynb`: Notebook principal con todo el flujo de procesamiento, vectorización e indexación.
- `pdfs/`: Carpeta donde colocar los archivos PDF a procesar. Incluye un PDF de ejemplo: **MS-15B1_v1.0_English.pdf** (manual de notebook).
- `poppler/` y `tesseract/`: Incluyen instaladores y binarios necesarios para manipular PDFs y realizar OCR en Windows.
- `faiss_index.bin`, `embeddings.npy`, `chunks.pkl`: Archivos generados tras el procesamiento, contienen el índice, los embeddings y los textos segmentados.

## Requisitos
- Python 3.8+
- Dependencias: ver la primera celda del notebook para instalación automática (`pip install ...`).
- Windows (se incluyen binarios de Poppler y Tesseract para facilitar la ejecución).

## Cómo usar
1. Coloca tus archivos PDF en la carpeta `pdfs/`.
2. Abre y ejecuta el notebook `Notebook.ipynb` en VS Code o Jupyter.
3. El flujo realiza:
	- Extracción de texto (OCR si es necesario)
	- Segmentación en chunks
	- Generación de embeddings
	- Indexación con FAISS
	- Ejemplo de consulta y respuesta
4. Los resultados se guardan automáticamente para uso posterior.

## Herramientas incluidas
- **PDF de ejemplo:** `pdfs/MS-15B1_v1.0_English.pdf` (manual de notebook para pruebas).
- **Poppler:** Binarios en `poppler/bin/` para manipulación de PDFs (requerido por `pdf2image`).
- **Tesseract:** Binarios y ejecutable en `tesseract/` para OCR de imágenes.

No es necesario instalar Poppler ni Tesseract por separado: el notebook ya está configurado para usar los binarios incluidos.

## Ejemplo de consulta y respuesta

> **Pregunta:** ¿Cómo es el unpacking del producto?
>
> **Respuesta del asistente:**
> El unpacking del producto se realiza de la siguiente manera: primero, debes desempacar la caja de envío y revisar cuidadosamente todos los artículos incluidos. Si encuentras algún artículo dañado o faltante, debes contactar a tu distribuidor local de inmediato. Además, se recomienda guardar la caja y los materiales de embalaje por si necesitas enviar el producto en el futuro. El paquete debe contener los siguientes elementos: un notebook, una guía de inicio rápido, un adaptador AC/DC y un cable de alimentación AC, y una bolsa de transporte opcional.