import os
import base64
import streamlit as st
from pathlib import Path
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_milvus import Milvus
import fitz  # PyMuPDF for handling PDFs
import tempfile
import markdown2
import pdfkit
from PIL import Image, ImageDraw
import hashlib
import uuid
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, Index, utility
import atexit

# Set the API key using st.secrets for secure access
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
MODEL = "gpt-4o"
client = OpenAI()
embeddings = OpenAIEmbeddings()

# Milvus connection parameters
MILVUS_ENDPOINT = st.secrets["general"]["MILVUS_PUBLIC_ENDPOINT"]
MILVUS_API_KEY = st.secrets["general"]["MILVUS_API_KEY"]
MILVUS_CONNECTION_ARGS = {
    "uri": MILVUS_ENDPOINT,
    "token": MILVUS_API_KEY,
    "secure": True
}

def connect_to_milvus():
    try:
        connections.connect(alias="default", **MILVUS_CONNECTION_ARGS)
        st.success("Connected to Milvus successfully!")
    except Exception as e:
        st.error(f"Failed to connect to Milvus: {str(e)}")
        raise

def close_milvus_connection():
    try:
        connections.disconnect(alias="default")
        st.success("Disconnected from Milvus successfully!")
    except Exception as e:
        st.error(f"Failed to disconnect from Milvus: {str(e)}")

atexit.register(close_milvus_connection)

def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

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

def generate_summary(content):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
            {"role": "user", "content": f"Provide a brief summary of this document, including main topics and key points:\n\n{content}"}
        ]
    )
    return response.choices[0].message.content

def calculate_confidence(docs):
    return min(len(docs) / 5 * 100, 100)  # 5 is the max number of chunks we retrieve

def highlight_relevant_text(text, query):
    highlighted = text.replace(query, f"**{query}**")
    return highlighted

def process_file(uploaded_file):
    st.subheader(f"Processing: {uploaded_file.name}")
    
    progress_bar = st.progress(0)
    
    temp_file_path = save_uploadedfile(uploaded_file)
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension == '.pdf':
        # Process PDF
        doc = fitz.open(temp_file_path)
        output_dir = tempfile.mkdtemp()

        total_pages = len(doc)
        image_paths = []
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            output = os.path.join(output_dir, f"page{page_num + 1}.png")
            pix.save(output)
            image_paths.append((page_num + 1, output))
            progress_bar.progress((page_num + 1) / total_pages)

        doc.close()
        st.success('PDF converted to images successfully!')
    elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
        # Process single image
        image_paths = [(1, temp_file_path)]
        progress_bar.progress(1.0)
        st.success('Image processed successfully!')
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return None, None, None, None

    markdown_content = ""
    for i, (page_num, img_path) in enumerate(image_paths):
        markdown_content += f"\n## Page {page_num}\n"
        markdown_content += get_generated_data(img_path)
        progress_bar.progress((i + 1) / len(image_paths))

    md_file_path = os.path.join(tempfile.mkdtemp(), "extracted_content.md")
    with open(md_file_path, "w") as f:
        f.write(markdown_content)

    loader = UnstructuredMarkdownLoader(md_file_path)
    data = loader.load()

    for i, page in enumerate(data):
        page.metadata['page_number'] = i + 1
        file_hash = get_file_hash(page.page_content.encode())
        page.metadata['file_hash'] = file_hash

    vector_db = Milvus.from_documents(
        data,
        embeddings,
        connection_args=MILVUS_CONNECTION_ARGS,
    )

    summary = generate_summary(markdown_content)

    progress_bar.progress(100)

    return vector_db, image_paths, markdown_content, summary

def create_document_vectors_schema():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="file_hash", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="page_number", dtype=DataType.INT64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)  # Adjust dim based on your embedding size
    ]
    schema = CollectionSchema(fields, "Document vectors collection")
    return schema

def create_index(collection, field_name="embedding"):
    # Define the index parameters
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name=field_name, params=index_params)
    st.success(f"Index created on field '{field_name}' for collection '{collection.name}'.")

def get_or_create_collection(collection_name):
    try:
        collection = Collection(collection_name)
        collection.load()
        if not collection.has_index():  # Explicitly check if the index exists
            create_index(collection)  # Call to create the index if it does not exist
        return collection
    except Exception as e:
        if 'does not exist' in str(e):  # Better error message handling
            collection_schema = create_document_vectors_schema()
            collection = Collection(name=collection_name, schema=collection_schema)
            collection.create()
            create_index(collection)  # Ensure index is created right after the collection
            st.success(f"Collection '{collection_name}' created with an index.")
            return collection
        else:
            raise

# Initialize Milvus connection and collections
connect_to_milvus()
document_collection = get_or_create_collection("LangChainCollection")

# Streamlit interface
st.title('Document Query and Analysis App')

with st.expander("Legal Disclaimer"):
    st.markdown("""
    **Legal Disclaimer:**
    - This application is for entertainment purposes only.
    - The data processed by this app is not stored permanently and is only used for the current session.
    - No warranties or guarantees are provided regarding the accuracy or reliability of the results generated by this app.
    - By using this application, you agree that you will not hold the developers or any affiliated parties liable for any damages or losses that may arise from its use.
    - Ensure that you do not upload any sensitive or confidential information, as the data will be processed by external services.
    """)

try:
    # Initialize session state variables
    if 'current_session_files' not in st.session_state:
        st.session_state['current_session_files'] = set()
    if 'processed_data' not in st.session_state:
        st.session_state['processed_data'] = {}
    if 'file_hashes' not in st.session_state:
        st.session_state['file_hashes'] = {}

    # Sidebar for advanced options
    with st.sidebar:
        st.header("Advanced Options")
        chunks_to_retrieve = st.slider("Number of chunks to retrieve", 1, 10, 5)
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5)

        if st.button("Clear Current Session"):
            st.session_state['current_session_files'] = set()
            st.session_state['file_hashes'] = {}
            st.success("Current session cleared. You can now upload new files.")

    uploaded_files = st.file_uploader("Upload PDF or Image file(s)", type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_content = uploaded_file.getvalue()
            file_hash = get_file_hash(file_content)
            
            if file_hash in st.session_state['file_hashes']:
                # File has been processed before
                existing_file_name = st.session_state['file_hashes'][file_hash]
                st.session_state['current_session_files'].add(existing_file_name)
                st.success(f"File '{uploaded_file.name}' has already been processed as '{existing_file_name}'. Using existing data.")
            else:
                # New file, needs processing
                try:
                    vector_db, image_paths, markdown_content, summary = process_file(uploaded_file)
                    if vector_db is not None:
                        st.session_state['processed_data'][uploaded_file.name] = {
                            'vector_db': vector_db,
                            'image_paths': image_paths,
                            'markdown_content': markdown_content,
                            'summary': summary
                        }
                        st.session_state['current_session_files'].add(uploaded_file.name)
                        st.session_state['file_hashes'][file_hash] = uploaded_file.name

                        # Insert into document collection
                        document_collection.insert([
                            {
                                "file_hash": file_hash,
                                "page_number": page.metadata['page_number'],
                                "content": page.page_content,
                                "embedding": page.embedding
                            }
                            for page in data
                        ])

                        st.success(f"File processed and stored in vector database! Summary: {summary}")

            # Display summary and extracted content
            display_name = uploaded_file.name if uploaded_file.name in st.session_state['processed_data'] else st.session_state['file_hashes'].get(file_hash, uploaded_file.name)
            with st.expander(f"View Summary for {display_name}"):
                st.markdown(st.session_state['processed_data'][display_name]['summary'])
            with st.expander(f"View Extracted Content for {display_name}"):
                st.markdown(st.session_state['processed_data'][display_name]['markdown_content'])

    # Display all uploaded images for the current session
    if st.session_state['current_session_files']:
        st.subheader("Uploaded Documents and Images")
        for file_name in st.session_state['current_session_files']:
            with st.expander(f"Images from {file_name}"):
                for page_num, image_path in st.session_state['processed_data'][file_name]['image_paths']:
                    st.image(image_path, caption=f"Page {page_num}", use_column_width=True)

    # Query interface
    st.subheader("Query the Document(s)")
    query = st.text_input("Enter your query about the document(s):")
    file_content = st.file_uploader("Upload the same file to access the content:", type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif"])
    if st.button("Search"):
        if file_content:
            file_hash = get_file_hash(file_content.getvalue())
            results = document_collection.query(
                expr=f'file_hash == "{file_hash}"',
                output_fields=["file_name", "page_number", "content", "embedding"]
            )
            if results:
                with st.spinner('Searching...'):
                    all_docs = []
                    for result in results:
                        vector_db = st.session_state['processed_data'][result['file_name']]['vector_db']
                        docs = vector_db.similarity_search(query, k=chunks_to_retrieve)
                        all_docs.extend([(result['file_name'], doc) for doc in docs])
                    
                    if all_docs:
                        content = "\n".join([f"File: {file_name}, Page {doc.metadata.get('page_number', 'Unknown')}: {doc.page_content}" for file_name, doc in all_docs])

                        system_content = "You are an assisting agent. Please provide the response based on the input. After your response, list the sources of information used, including file names, page numbers, and relevant snippets."
                        user_content = f"Respond to the query '{query}' using the information from the following content: {content}"

                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": system_content},
                                {"role": "user", "content": user_content}
                            ]
                        )
                        
                        st.subheader("Answer:")
                        st.write(response.choices[0].message.content)

                        confidence_score = calculate_confidence(all_docs)
                        st.write(f"Confidence Score: {confidence_score}%")

                        st.subheader("Sources:")
                        for file_name, doc in all_docs:
                            page_num = doc.metadata.get('page_number', 'Unknown')
                            st.markdown(f"**File: {file_name}, Page {page_num}:**")
                            highlighted_text = highlight_relevant_text(doc.page_content[:200], query)
                            st.markdown(f"```\n{highlighted_text}...\n```")
                            
                            image_path = next((img_path for num, img_path in st.session_state['processed_data'][file_name]['image_paths'] if num == page_num), None)
                            if image_path:
                                with st.expander(f"View Page {page_num} Image"):
                                    st.image(image_path, use_column_width=True)
                    else:
                        st.error("No results found or file does not match.")
            else:
                st.error("No documents found for the uploaded file.")
        else:
            st.error("Please upload a file to verify access.")

except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    st.exception(e)

# Ensure Milvus connection is closed when the script ends
atexit.register(close_milvus_connection)
