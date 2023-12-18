from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from io import BytesIO
from langchain.document_loaders import PyPDFLoader
import gradio as gr


local_llm = "zephyr-7b-alpha.Q5_K_S.gguf"

config = {
    'max_new_tokens': 1024,
    'repetition_penalty': 1.1,
    'temperature': 0.1,
    'top_k': 50,
    'top_p': 0.9,
    'stream': True,
    'threads': int(os.cpu_count() / 2)
}

llm = CTransformers(
    model=local_llm,
    model_type="mistral",
    lib="avx2",  # for CPU use
    **config
)

print("LLM Initialized...")


prompt_template = """Utiliza la siguiente información para responder a la pregunta del usuario.
Si no sabes la respuesta, di simplemente que no la sabes, no intentes inventarte una respuesta.

Contexto: {context}
Pregunta: {question}

Devuelve sólo la respuesta útil que aparece a continuación y nada más.
Responde siempre en castellano
Respuesta útil:
"""

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question'])
load_vector_store = Chroma(
    persist_directory="stores/pet_cosine", embedding_function=embeddings)
retriever = load_vector_store.as_retriever(search_kwargs={"k": 1})
# query = "what is the fastest speed for a greyhound dog?"
# semantic_search = retriever.get_relevant_documents(query)
# print(semantic_search)

print("######################################################################")

chain_type_kwargs = {"prompt": prompt}

# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents = True,
#     chain_type_kwargs= chain_type_kwargs,
#     verbose=True
# )

# response = qa(query)

# print(response)

sample_prompts = ["En caso de empate entre el alumnado de alguna especialidad de la enseñanza profesionales de música, ¿Qué criterios se aplicarían para dar el premio?",
                  "¿Qué requisitos debe reunir un alumno candidato al premio extraordinario de enseñanzas profesionales de música?", "¿Cuál es la fecha de publicación en el BOE de la Orden ECD/1611/2015, del 29 de julio, del Ministerio de Educación, Cultura y Deporte?"]


def get_response(input):
    query = input
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
    response = qa(query)
    return response["result"]


input = gr.Text(
    label="Prompt",
    show_label=False,
    max_lines=1,
    placeholder="Enter your prompt",
    container=False,
)

iface = gr.Interface(fn=get_response,
                     inputs=input,
                     outputs="text",
                     title="ConserGPT",
                     description="This is a RAG implementation based on Zephyr 7B Alpha LLM.",
                     examples=sample_prompts,
                     allow_flagging=False,
                     share=True
                     )

iface.launch()
