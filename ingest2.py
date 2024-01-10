import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

pdf_folder = "./pdf_folder/pdf"  # Ruta a la carpeta que contiene los archivos PDF
output_folder = "stores/ConserGPT"  # Carpeta de salida para los vector stores

# Crear el directorio de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Iterar a travÃ©s de los archivos PDF en la carpeta
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, pdf_file)

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        vector_store = Chroma.from_documents(texts, embeddings, collection_metadata={
                                             "hnsw:space": "cosine"}, persist_directory=os.path.join(output_folder, f"{pdf_file}_store"))

        print(f"Vector Store created for {pdf_file}")

print("All Vector Stores Created.......")
