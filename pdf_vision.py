import os
import base64
import streamlit as st
from pathlib import Path
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_milvus.vectorstores import Milvus
import fitz  # PyMuPDF for handling PDFs
import tempfile
import logging
from pymilvus import connections, Collection

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Set the API key using st.secrets for secure access
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
MODEL = "gpt-4o"
client = OpenAI()
embeddings = OpenAIEmbeddings()

# Milvus connection parameters from st.secrets
MILVUS_ENDPOINT = st.secrets["general"]["MILVUS_PUBLIC_ENDPOINT"]
MILVUS_API_KEY = st.secrets["general"]["MILVUS_API_KEY"]

# Log the connection parameters
logging.debug(f"Connecting to Milvus server at {MILVUS_ENDPOINT} with API key.")

# Connect to Milvus server
connections.connect("default", uri=MILVUS_ENDPOINT, token=MILVUS_API_KEY, secure=True)

# Check if the collection exists
collection_name = "pdf_embeddings"
try:
    collection = Collection(name=collection_name)
    if collection.name:
        st.write(f"Collection '{collection_name}' is available.")
except Exception as e:
    st.error(f"Collection '{collection_name}' does not exist or could not be accessed: {str(e)}")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

SYSTEM_PROMPT = "You are a helpful assistant that responds in Markdown. Help me with Given Image Extraction with Given Details with Different categories!"
USER_PROMPT = """
Retrieve all the information provided in the image, including figures, titles, and graphs.
"""

def get_generated_data(image_path):
    base64_image = encode_image(image_path)
    
    response = client.chat_completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

def save_uploadedfile(uploadedfile):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

st.title('PDF Document Query and Analysis App')

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    temp_file_path = save_uploadedfile(uploaded_file)
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()

    # Create and store embeddings in Milvus
    try:
        vector_db = Milvus.from_documents(
            pages,
            embeddings,
            connection_args={"alias": "default"},
        )
        st.session_state['vector_db'] = vector_db
        st.success('Embeddings stored successfully in Milvus!')
    except Exception as e:
        logging.error(f"Failed to create connection to Milvus server: {e}")
        st.error("Failed to connect to the Milvus server. Please check the connection parameters and try again.")

    # Process for querying embeddings
    query = st.text_input("Enter your query for the PDF document:")
    if st.button("Query PDF"):
        if 'vector_db' in st.session_state:
            docs = st.session_state['vector_db'].similarity_search(query)
            content = "\n".join(doc.page_content for doc in docs)

            system_content = "You are a helpful assistant. Provide the response based on the input."
            user_content = f"Answer the query '{query}' using the following content: {content}"

            response = client.chat_completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            )
            st.write(response.choices[0].message.content)
        else:
            st.error("Vector database is not available. Please upload a PDF and try again.")
else:
    st.write("Please upload a PDF to begin processing.")

# Further markdown processing if needed
if 'vector_db' in st.session_state:
    # Assuming we want to store markdown data as well
    markdown_content = ""
    folder_path = Path(tempfile.mkdtemp())  # Temporary folder for markdown files

    # Iterate over files in the directory and extract information
    for file_path in folder_path.iterdir():
        markdown_content += "\n" + get_generated_data(file_path)

    # Load markdown file
    loader = UnstructuredMarkdownLoader(folder_path)
    data = loader.load()

    # Save data into Milvus database
    try:
        vector_db = Milvus.from_documents(
            data,
            embeddings,
            connection_args={"alias": "default"},
        )
        st.session_state['vector_db'] = vector_db
    except Exception as e:
        logging.error(f"Failed to create connection to Milvus server: {e}")
        st.error("Failed to connect to the Milvus server. Please check the connection parameters and try again.")
