PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using Python, LangChain, ChromaDB, Hugging Face Embeddings, Google Gemini, and Gradio.

Features

- Upload and process PDF documents
- Split documents into chunks
- Generate embeddings using Hugging Face models
- Store embeddings in ChromaDB
- Retrieve relevant context from PDFs
- Answer questions using Google Gemini
- Simple Gradio web interface

Tech Stack

- Python
- LangChain
- ChromaDB
- Hugging Face Embeddings
- Google Gemini
- Gradio

Project Workflow

1. Load PDF documents
2. Split documents into chunks
3. Generate embeddings
4. Store embeddings in ChromaDB
5. Retrieve relevant chunks based on user query
6. Send retrieved context to Gemini
7. Generate an accurate answer

Installation

Clone the repository:

git clone <repository-url>
cd <repository-name>

Install dependencies:

pip install -r requirements.txt

Environment Variables

Create a ".env" file and add your Gemini API key:

MY_API_KEY=YOUR_GEMINI_API_KEY

Run the Application

python chat.py

Or if using Gradio:

python app.py

The application will start locally and provide a browser URL.

Example Questions

- What is corrosion?
- Explain galvanic corrosion.
- What are the types of corrosion?
- How can corrosion be prevented?

Future Improvements

- Multiple PDF support
- Chat history
- Source citations
- Persistent vector database
- Better UI and deployment

