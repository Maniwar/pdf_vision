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
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections
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
        connections.connect(**MILVUS_CONNECTION_ARGS)
        st.success("Connected to Milvus successfully!")
    except Exception as e:
        st.error(f"Failed to connect to Milvus: {str(e)}")
        raise

def close_milvus_connection():
    try:
        connections.disconnect("default")
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

def process_file(uploaded_file, session_key):
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
        page.metadata['session_key'] = session_key

    vector_db = Milvus.from_documents(
        data,
        embeddings,
        connection_args=MILVUS_CONNECTION_ARGS,
    )

    summary = generate_summary(markdown_content)

    progress_bar.progress(100)

    return vector_db, image_paths, markdown_content, summary

def create_session_collection():
    fields = [
        FieldSchema(name="session_key", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="file_hash", dtype=DataType.VARCHAR, max_length=64),
    ]
    schema = CollectionSchema(fields, "Session information collection")
    return schema

def get_or_create_collection(collection_name, schema=None):
    try:
        return Collection(collection_name)
    except Exception as e:
        if schema:
            return Collection(collection_name, schema)
        else:
            raise

# Initialize Milvus connection and collections
connect_to_milvus()
vector_collection = get_or_create_collection("document_vectors")
session_collection = get_or_create_collection("session_info", create_session_collection())

def generate_session_key():
    return hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()

# Streamlit interface
st.title('Document Query and Analysis App')

try:
    # Initialize session state variables
    if 'session_key' not in st.session_state:
        st.session_state['session_key'] = generate_session_key()
    if 'current_session_files' not in st.session_state:
        st.session_state['current_session_files'] = set()
    if 'processed_data' not in st.session_state:
        st.session_state['processed_data'] = {}
    if 'file_hashes' not in st.session_state:
        st.session_state['file_hashes'] = {}

    # Display current session key
    st.sidebar.subheader("Your Session Key")
    st.sidebar.code(st.session_state['session_key'])
    if st.sidebar.button("Copy Session Key"):
        st.sidebar.success("Session key copied to clipboard!")
        st.sidebar.text("Store this key securely to resume your session later.")

    # Option to enter a session key
    entered_key = st.sidebar.text_input("Enter a session key to resume:")
    if st.sidebar.button("Load Session"):
        results = session_collection.query(
            expr=f'session_key == "{entered_key}"',
            output_fields=["file_name", "file_hash"]
        )
        if results:
            st.session_state['session_key'] = entered_key
            st.session_state['current_session_files'] = set()
            st.session_state['file_hashes'] = {}
            for result in results:
                file_name = result['file_name']
                file_hash = result['file_hash']
                st.session_state['current_session_files'].add(file_name)
                st.session_state['file_hashes'][file_hash] = file_name
            st.success("Session loaded successfully!")
        else:
            st.error("Invalid session key. Please try again.")

    # Sidebar for advanced options
    with st.sidebar:
        st.header("Advanced Options")
        chunks_to_retrieve = st.slider("Number of chunks to retrieve", 1, 10, 5)
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5)

        if st.button("Clear Current Session"):
            new_key = generate_session_key()
            st.session_state['session_key'] = new_key
            st.session_state['current_session_files'] = set()
            st.session_state['processed_data'] = {}
            st.session_state['file_hashes'] = {}
            st.success("Current session cleared. A new session key has been generated.")

    uploaded_files = st.file_uploader("Upload PDF or Image file(s)", type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_content = uploaded_file.getvalue()
            file_hash = get_file_hash(file_content)
            
            # Check if file exists in the current session
            results = session_collection.query(
                expr=f'session_key == "{st.session_state["session_key"]}" and file_hash == "{file_hash}"',
                output_fields=["file_name"]
            )
            
            if results:
                # File has been processed before in this session
                existing_file_name = results[0]['file_name']
                st.session_state['current_session_files'].add(existing_file_name)
                st.success(f"File '{uploaded_file.name}' has already been processed as '{existing_file_name}'. Using existing data.")
            else:
                # New file, needs processing
                try:
                    vector_db, image_paths, markdown_content, summary = process_file(uploaded_file, st.session_state["session_key"])
                    if vector_db is not None:
                        st.session_state['processed_data'][uploaded_file.name] = {
                            'vector_db': vector_db,
                            'image_paths': image_paths,
                            'markdown_content': markdown_content,
                            'summary': summary
                        }
                        st.session_state['current_session_files'].add(uploaded_file.name)
                        st.session_state['file_hashes'][file_hash] = uploaded_file.name
                        
                        # Store session information
                        session_collection.insert([
                            [st.session_state["session_key"]],
                            [uploaded_file.name],
                            [file_hash]
                        ])
                        
                        st.success(f"File processed and stored in vector database! Summary: {summary}")
                except Exception as e:
                    st.error(f"An error occurred while processing {uploaded_file.name}: {str(e)}")

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
    if st.button("Search"):
        if st.session_state['current_session_files']:
            with st.spinner('Searching...'):
                all_docs = []
                for file_name in st.session_state['current_session_files']:
                    vector_db = st.session_state['processed_data'][file_name]['vector_db']
                    docs = vector_db.similarity_search(
                        query, 
                        k=chunks_to_retrieve,
                        filter=f'session_key == "{st.session_state["session_key"]}"'
                    )
                    all_docs.extend([(file_name, doc) for doc in docs if doc.metadata.get('page_number') is not None])
                
                # Sort all_docs by relevance
                all_docs.sort(key=lambda x: x[1].metadata.get('relevance', 0), reverse=True)
                
                content = "\n".join([f"File: {file_name}, Page {doc.metadata['page_number']}: {doc.page_content}" for file_name, doc in all_docs])

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
                    page_num = doc.metadata['page_number']
                    st.markdown(f"**File: {file_name}, Page {page_num}:**")
                    highlighted_text = highlight_relevant_text(doc.page_content[:200], query)
                    st.markdown(f"```\n{highlighted_text}...\n```")
                    image_path = next((img_path for num, img_path in st.session_state['processed_data'][file_name]['image_paths'] if num == page_num), None)
                    if image_path:
                        with st.expander(f"View Page {page_num} Image"):
                            st.image(image_path, use_column_width=True)

            # Save question and answer to history
            if 'qa_history' not in st.session_state:
                st.session_state['qa_history'] = []
            st.session_state['qa_history'].append({
                'question': query,
                'answer': response.choices[0].message.content,
                'sources': [{'file': file_name, 'page': doc.metadata['page_number']} for file_name, doc in all_docs],
                'confidence': confidence_score
            })

        else:
            st.warning("Please upload and process at least one file first.")

    # Display question history
    if 'qa_history' in st.session_state and st.session_state['qa_history']:
        st.subheader("Question History")
        for i, qa in enumerate(st.session_state['qa_history']):
            with st.expander(f"Q{i+1}: {qa['question']}"):
                st.write(f"A: {qa['answer']}")
                st.write(f"Confidence: {qa['confidence']}%")
                st.write("Sources:")
                for source in qa['sources']:
                    st.write(f"- File: {source['file']}, Page: {source['page']}")
        
        # Add a button to clear the question history
        if st.button("Clear Question History"):
            st.session_state['qa_history'] = []
            st.success("Question history cleared!")

    # Export results
    if st.button("Export Q&A Session"):
        qa_session = ""
        for qa in st.session_state.get('qa_history', []):
            qa_session += f"Q: {qa['question']}\n\nA: {qa['answer']}\n\nConfidence: {qa['confidence']}%\n\nSources:\n"
            for source in qa['sources']:
                qa_session += f"- File: {source['file']}, Page: {source['page']}\n"
            qa_session += "\n---\n\n"
        
        # Convert markdown to HTML
        html = markdown2.markdown(qa_session)
        
        try:
            # Convert HTML to PDF
            pdf = pdfkit.from_string(html, False)
            
            # Provide the PDF for download
            st.download_button(
                label="Download Q&A Session as PDF",
                data=pdf,
                file_name="qa_session.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"An error occurred while generating the PDF: {str(e)}")

except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    st.exception(e)

# Ensure Milvus connection is closed when the script ends
atexit.register(close_milvus_connection)
