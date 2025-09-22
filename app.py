# app.py
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import requests

# --- FINAL, CORRECTED IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
# UPDATED: Import for the modern HuggingFace embeddings class
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables.
load_dotenv()

# Configure Flask app
app = Flask(__name__)

# Global variable to hold our knowledge base
vector_store = None

# --- Translation Helper Function for LibreTranslate ---
def translate_text(text, target_language, source_language='auto'):
    if not text: return ""
    api_url = "http://localhost:5000/translate"
    payload = {"q": text, "source": source_language, "target": target_language, "format": "text"}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("translatedText", text)
    except Exception as e:
        print(f"Error during translation with LibreTranslate: {e}")
        return text

# --- Main Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global vector_store
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No PDF file provided."}), 400
    
    file = request.files['pdf_file']
    if file.filename == '' or not file.filename.endswith('.pdf'):
        return jsonify({"error": "Please select a valid PDF file."}), 400

    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        filepath = os.path.join(temp_dir, file.filename)
        file.save(filepath)

        loader = PyPDFLoader(filepath)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        print("Loading local embedding model (this may take a while)...")
        # UPDATED: Use the modern, non-deprecated class
        embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
        print("Model loaded. Creating vector store...")
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)
        print("Vector store created successfully.")

        os.remove(filepath)
        return jsonify({"success": f"PDF '{file.filename}' is ready."})

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return jsonify({"error": "Failed to process PDF."}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message")
    lang_code = data.get("language", "en")

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    try:
        message_in_english = translate_text(user_message, 'en', source_language=lang_code)
        response_in_english = ""

        # UPDATED: Use a modern, stable model name like 'gemini-1.5-flash-latest'
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

        if vector_store:
            docs = vector_store.similarity_search(message_in_english, k=3)
            chain = load_qa_chain(llm, chain_type="stuff")
            response_dict = chain.invoke({"input_documents": docs, "question": message_in_english})
            response_in_english = response_dict.get('output_text', 'Could not process the response.')
        else:
            response = llm.invoke(message_in_english)
            response_in_english = response.content

        final_response = translate_text(response_in_english, lang_code, source_language='en')
        return jsonify({"reply": final_response})

    except Exception as e:
        print(f"Error during API call: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)