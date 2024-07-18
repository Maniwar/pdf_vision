import os
import base64
import streamlit as st
from pathlib import Path
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_milvus.vectorstores import Milvus
import fitz  # PyMuPDF for handling PDFs

# Set the API key using st.secrets for secure access
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
MODEL = "gpt-4o"
client = OpenAI()
embeddings = OpenAIEmbeddings()

# Milvus connection parameters from st.secrets
MILVUS_ENDPOINT = st.secrets["general"]["MILVUS_PUBLIC_ENDPOINT"]
MILVUS_API_KEY = st.secrets["general"]["MILVUS_API_KEY"]
MILVUS_CONNECTION_ARGS = {
    "host": MILVUS_ENDPOINT,
    "api_key": MILVUS_API_KEY,
}

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
    with open(os.path.join("tempDir", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return os.path.join("tempDir", uploadedfile.name)

# Streamlit interface
st.title('PDF Document Query and Analysis App')

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    st.subheader("PDF Processing and Image Extraction")
    temp_file_path = save_uploadedfile(uploaded_file)
    
    doc = fitz.open(temp_file_path)
    output_dir = "./data/output/"
    os.makedirs(output_dir, exist_ok=True)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        output = os.path.join(output_dir, f"page{page_num + 1}.png")
        pix.save(output)
    doc.close()

    st.success('PDF converted to images successfully!')

    # Process each image for data extraction
    folder_path = Path(output_dir)
    markdown_content = ""
    for file_path in folder_path.iterdir():
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

    # Define the URI for local storage
    URI = "./db/pdf.db"

    # Create and store embeddings in Milvus
    vector_db = Milvus.from_documents(
        pages,
        embeddings,
        connection_args={"uri": URI},
    )
    st.session_state['data'] = vector_db

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
