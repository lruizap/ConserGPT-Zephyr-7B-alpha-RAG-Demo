import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

txt_folder = "./pdf_folder/txt"  # Ruta a la carpeta que contiene los archivos TXT
output_folder = "stores/ConserGPT"  # Carpeta de salida para los vector stores

# Crear el directorio de salida si no existe
os.makedirs(output_folder, exist_ok=True)


def longitud_palabras_en_array(nombre_archivo):
    try:
        with open(nombre_archivo, 'r') as archivo:
            contenido = archivo.read()
            palabras = contenido.split()

            if not palabras:
                print(f"El archivo '{nombre_archivo}' está vacío.")
                return []

            longitudes = [len(palabra) for palabra in palabras]
            return longitudes
    except FileNotFoundError:
        print(f"El archivo '{nombre_archivo}' no fue encontrado.")
        return []
    except Exception as e:
        print(f"Ocurrió un error: {e}")
        return []


# Iterar a través de los archivos TXT en la carpeta
for txt_file in os.listdir(txt_folder):
    if txt_file.endswith(".txt"):
        txt_path = os.path.join(txt_folder, txt_file)

        txt_path = txt_path.replace('\\', '/')

        with open(txt_path, 'r', encoding='utf-8') as file:
            content = file.read()

        palabras = content.split(' ')

        # for palabra in range(0, len(palabras) - 1):
        #     text_splitter = CharacterTextSplitter(
        #         chunk_size=palabra, chunk_overlap=(palabra - 2), separator='')
        #     texts = text_splitter.split_text(content)

        vector_store = Chroma.from_texts(palabras, embeddings, collection_metadata={
            "hnsw:space": "cosine"}, persist_directory=os.path.join(output_folder, f"{txt_file}_store"))

        print(f"Vector Store created for {txt_file}")

print("All Vector Stores Created.......")
