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

# Extract and set Milvus connection parameters
MILVUS_ENDPOINT = st.secrets["general"]["MILVUS_PUBLIC_ENDPOINT"]
MILVUS_API_KEY = st.secrets["general"]["MILVUS_API_KEY"]
host_port = MILVUS_ENDPOINT.split("//")[-1]
host, port = (host_port.split(":") + [19530])[:2]  # Default port is 19530
port = int(port)
MILVUS_CONNECTION_ARGS = {"host": host, "port": port, "api_key": MILVUS_API_KEY, "secure": True}

# Log connection parameters
logging.debug(f"Connecting to Milvus server at {host}:{port} with API key.")

# Function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Define prompts
SYSTEM_PROMPT = "You are a helpful assistant that responds in Markdown. Help me with Given Image Extraction with Given Details with Different categories!"
USER_PROMPT = "Retrieve all the information provided in the image, including figures, titles, and graphs."

# Initialize Streamlit UI
st.title('PDF Document Query and Analysis App')
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Process PDF and display images
if uploaded_file:
    if 'output_dir' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        with st.spinner('Processing PDF...'):
            temp_file_path = save_uploadedfile(uploaded_file)
            doc = fitz.open(temp_file_path)
            output_dir = tempfile.mkdtemp()
            st.session_state.output_dir = output_dir

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                output = os.path.join(output_dir, f"page{page_num + 1}.png")
                pix.save(output)
            doc.close()
            st.success('PDF converted to images successfully!')

    if 'data' not in st.session_state:
        process_markdown_to_embeddings(st.session_state.output_dir)

    display_pdf_images_and_content()

def save_uploadedfile(uploadedfile):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

def process_markdown_to_embeddings(output_dir):
    markdown_content = ""
    image_files = list(Path(output_dir).iterdir())
    for file_path in sorted(image_files):
        markdown_content += "\n" + get_generated_data(str(file_path))

    temp_md_file = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
    with open(temp_md_file.name, 'w') as f:
        f.write(markdown_content)

    loader = UnstructuredMarkdownLoader(temp_md_file.name)
    data = loader.load()
    try:
        vector_db = Milvus.from_documents(data, embeddings, connection_args=MILVUS_CONNECTION_ARGS)
        st.session_state['data'] = vector_db
    except Exception as e:
        logging.error(f"Failed to connect to Milvus server: {e}")
        st.error(f"Connection error: {str(e)}")

def display_pdf_images_and_content():
    # Display the query interface and extracted markdown content above the images
    query = st.text_input("Enter your query about the PDF content:")
    if st.button("Query PDF"):
        if 'data' in st.session_state and st.session_state['data']:
            vector_db = st.session_state['data']
            docs = vector_db.similarity_search(query)
            content = "\n".join(doc.page_content for doc in docs)
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": "You are a helpful assistant. Provide the response based on the input."},
                          {"role": "user", "content": f"Answer the query '{query}' using the following content: {content}"}],
                temperature=0.0
            )
            st.write(response.choices[0].message.content)
        else:
            st.error("Please upload a PDF first and wait for the embeddings to be processed.")

    # Display images from PDF
    st.subheader("Generated Images from PDF")
    folder_path = Path(st.session_state.output_dir)
    image_files = list(folder_path.iterdir())
    for image_file in sorted(image_files):
        st.image(str(image_file), caption=f"Page {image_file.stem.split('page')[1]}")
