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

# Extract the host and port from the endpoint
host_port = MILVUS_ENDPOINT.split("//")[-1]
if ":" in host_port:
    host, port = host_port.split(":")
    port = int(port)
else:
    host = host_port
    port = 19530  # Default port

MILVUS_CONNECTION_ARGS = {
    "host": host,
    "port": port,
    "api_key": MILVUS_API_KEY,
    "secure": True  # Ensure the connection uses HTTPS
}

# Log the connection parameters
logging.debug(f"Connecting to Milvus server at {host}:{port} with API key.")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

SYSTEM_PROMPT = "You are a helpful assistant that responds in Markdown. Help me with Given Image Extraction with Given Details with Different categories!"
USER_PROMPT = """
Retrieve all the information provided in the image, including figures, titles, and graphs.
"""

def get_generated_data(image_path):
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(
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

# Streamlit interface
st.title('PDF Document Query and Analysis App')

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    st.subheader("PDF Processing and Image Extraction")
    temp_file_path = save_uploadedfile(uploaded_file)
    
    doc = fitz.open(temp_file_path)
    output_dir = tempfile.mkdtemp()

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        output = os.path.join(output_dir, f"page{page_num + 1}.png")
        pix.save(output)
    doc.close()

    st.success('PDF converted to images successfully!')

    # Display the generated images
    st.subheader("Generated Images from PDF")
    folder_path = Path(output_dir)
    image_files = list(folder_path.iterdir())
    for image_file in sorted(image_files):
        st.image(str(image_file), caption=f"Page {image_file.stem.split('page')[1]}")

    # Process each image for data extraction
    markdown_content = ""
    for file_path in image_files:
        markdown_content += "\n" + get_generated_data(str(file_path))

    # Display extracted markdown content
    st.text_area("Extracted Content:", markdown_content, height=300)

    # Query interface based on extracted content
    query = st.text_input("Enter your query about the extracted data:")
    if st.button("Search"):
        with st.spinner('Searching...'):
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an assisting agent. Please provide the response based on the input."},
                    {"role": "user", "content": f"Respond to the query '{query}' using the information from the following content: {markdown_content}"}
                ]
            )
            st.write(response.choices[0].message.content)

# For embedding and storing in Milvus
if 'data' not in st.session_state:
    st.session_state['data'] = []

def process_pdfs_to_embeddings(uploaded_file):
    temp_file_path = save_uploadedfile(uploaded_file)
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()

    # Create and store embeddings in Milvus
    try:
        vector_db = Milvus.from_documents(
            pages,
            embeddings,
            connection_args=MILVUS_CONNECTION_ARGS,
        )
        st.session_state['data'] = vector_db
    except Exception as e:
        logging.error(f"Failed to create connection to Milvus server: {e}")
        st.error("Failed to connect to the Milvus server. Please check the connection parameters and try again.")

if uploaded_file:
    process_pdfs_to_embeddings(uploaded_file)

query = st.text_input("Enter your query for the PDF document:")
if st.button("Query PDF"):
    vector_db = st.session_state.get('data')
    if vector_db:
        docs = vector_db.similarity_search(query)
        content = ""
        for doc in docs:
            content += "\n" + doc.page_content

        system_content = "You are a helpful assistant. Provide the response based on the input."
        user_content = f"Answer the {query} from the {content}"

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
        )
        st.write(response.choices[0].message.content)
    else:
        st.error("Please upload a PDF first.")
