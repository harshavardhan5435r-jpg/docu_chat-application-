
#step 1:" we need to load our pdf " 
#install dependencies (!pip install -qU langchain-community pypdf)

from langchain_community.document_loaders import PyPDFLoader

# 1. Provide the local file path to your PDF
loader = PyPDFLoader(r"C:\Users\HARSHA\Downloads\Corrosion Notes PDF 01-12-2025.pdf")
# 2. Load the document (splits the PDF page-by-page automatically)
document  = loader.load()
# 3. View the extracted content from the first page
#print(document[0].page_content) so output will be first page_content


#step 2:" We need to split the pages into chunks "
#install dependencies(!pip install -qU langchain-community pypdf)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_split= RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)

splitted=text_split.split_documents(document)

#print(len(splitted)) for getting no of chunks  the document is splitted.
#print(splitted[0].page_content) for getting the chunk with index 0 content.
#print(splitted[0].metadata) helps to retrieve metadata from the chunk with index 0.


#step 3: now we need to define embedding.
#install dependencies (!pip install -qU langchain-huggingface sentence-transformers)

from langchain_huggingface import HuggingFaceEmbeddings

# Initialize local model (it downloads the weights on the first run)
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

#step 4: now we need to store in vector databases so we can able to retrieve relevant information from document
# install dependencies(!pip install -qU langchain-chroma) 
from langchain_chroma import Chroma
vector_store =Chroma(
    collection_name="research_collection",
    embedding_function=embeddings_model,
    persist_directory="./chroma_langchain_db"
)
documented=vector_store.add_documents(splitted)
#print(documented)



#step 5:we need retrieve and generate
def retrieve_context(query: str, k: int = 2):
    #similarity_search is used to retrive query related documents and k represents top k no os  chunks from the documents. 
    retrieved_docs = vector_store.similarity_search(query, k=k)
    docs_content = ""

    for doc in retrieved_docs:
        docs_content += f"Source: {doc.metadata}\n"
        docs_content += f"Content: {doc.page_content}\n"

    return docs_content, retrieved_docs
#step 6: now we need to add an model we can use google model,openAI model or any other model
# install dependencies !  (pip install langchain-google-genai)

from langchain.chat_models import init_chat_model
import os 
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('MY_API_KEY')


model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key=api_key,
)



def docu_chat(user_query):
    context, source_docs = retrieve_context(user_query, k=2)

    system_message = f"""
    You are a helpful chatbot.
    Use only the following pieces of context to answer the question.
    Don't makeup any new information: {context}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query}
    ]

    response = model.invoke(messages)

    return response.content,source_docs,context
    #print(docu_chat("what is corrosion"))



   #step 7: now we need to test our chatbot  so i want to deploy in hugging face with gradio as UI. 

import gradio as gr
def chatbot(user_query):
    answer, _, _ = docu_chat(user_query)
    return answer


demo = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about your PDF..."),
    outputs=gr.Textbox( label="Answer",placeholder="The answer will appear here...",lines=8),
    title="RAG PDF CHATBOT",
    description="Ask questions about your PDF document and get answers based on the content."
)

demo.launch(debug=True , share=True)