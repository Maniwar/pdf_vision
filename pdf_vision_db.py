import os
import base64
import streamlit as st
from pathlib import Path
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import tempfile
import markdown2
import pdfkit
import hashlib
import tiktoken
import math
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
import pandas as pd
import io
import fitz
from docx import Document
from PIL import Image, ImageDraw, ImageFont
import textwrap
import mammoth
from pyvirtualdisplay import Display
import imgkit
import plotly.graph_objs as go

# Set page configuration to wide mode
st.set_page_config(layout="wide")

# Set the API key using st.secrets for secure access
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini"  # Latest GPT-4 Turbo model
MAX_TOKENS = 12000
client = OpenAI()
embeddings = OpenAIEmbeddings()

# Milvus connection parameters
MILVUS_ENDPOINT = st.secrets["general"]["MILVUS_PUBLIC_ENDPOINT"]
MILVUS_API_KEY = st.secrets["general"]["MILVUS_API_KEY"]

# iOS-like CSS styling
st.markdown("""
<style>
    /* iOS-like color palette */
    :root {
        --ios-blue: #007AFF;
        --ios-gray: #8E8E93;
        --ios-light-gray: #F2F2F7;
        --ios-white: #FFFFFF;
        --ios-red: #FF3B30;
    }

    /* General styling */
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
        color: #000000;
        background-color: var(--ios-light-gray);
    }

    /* Headings */
    h1, h2, h3 {
        font-weight: 600;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        background-color: var(--ios-blue);
        color: var(--ios-white);
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #0056b3;
    }

    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid var(--ios-gray);
        padding: 10px;
    }

    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: var(--ios-blue);
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: var(--ios-white);
        border-radius: 10px;
        border: 1px solid var(--ios-gray);
    }

    /* Warning banner */
    .warning-banner {
        background-color: #FFDAB9;
        border: 1px solid #FFA500;
        padding: 15px;
        color: #8B4513;
        font-weight: 600;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    /* Big font for important notices */
    .big-font {
        font-size: 24px !important;
        font-weight: 700;
        color: var(--ios-red);
    }

    /* Custom styling for alerts */
    .stAlert > div {
        padding: 15px;
        border-radius: 10px;
        display: flex;
        flex-direction: column;
        align-items: center;
        font-size: 16px;
    }

    .stAlert .big-font {
        margin-bottom: 10px;
    }

    .bottom-warning {
        background-color: #FFDDC1;
        border: 1px solid #FFA07A;
        padding: 15px;
        color: #8B0000;
        font-weight: 600;
        text-align: left;
        border-radius: 10px;
        margin-top: 20px;
    }

    .bottom-warning .big-font {
        font-size: 24px !important;
        font-weight: 700;
        color: #FF4500;
    }
</style>
""", unsafe_allow_html=True)

def connect_to_milvus():
    connections.connect(
        alias="default", 
        uri=MILVUS_ENDPOINT,
        token=MILVUS_API_KEY,
        secure=True
    )

def get_or_create_collection(collection_name, dim=1536):
    try:
        if utility.has_collection(collection_name):
            return Collection(collection_name)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=65535)
            ]
            schema = CollectionSchema(fields, "Document pages collection")
            collection = Collection(collection_name, schema)
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index("vector", index_params)
            return collection
    except Exception as e:
        st.error(f"Error in creating or accessing the collection: {str(e)}")
        return None

def toggle_content_visibility(key):
    if key not in st.session_state:
        st.session_state[key] = False
    st.session_state[key] = not st.session_state[key]


def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_all_documents():
    try:
        collection = get_or_create_collection("document_pages")
        if collection is None:
            return []
        collection.load()
        results = collection.query(
            expr="file_name != ''",
            output_fields=["file_name"],
            limit=16384
        )
        return list(set(doc['file_name'] for doc in results))
    except Exception as e:
        st.error(f"Error in fetching all documents: {str(e)}")
        return []

def get_document_content(file_name):
    if file_name in st.session_state.documents:
        return [
            {
                'content': content,
                'page_number': i + 1,
                'summary': st.session_state.documents[file_name]['summary']
            }
            for i, content in enumerate(st.session_state.documents[file_name]['page_contents'])
        ]

    try:
        collection = get_or_create_collection("document_pages")
        if collection is None:
            return []
        collection.load()
        results = collection.query(
            expr=f"file_name == '{file_name}'",
            output_fields=["content", "page_number", "summary"],
            limit=16384
        )
        return sorted(results, key=lambda x: x['page_number'])
    except Exception as e:
        st.error(f"Error in fetching document content: {str(e)}")
        return []

def calculate_confidence(score):
    # Convert the similarity score to a confidence level
    confidence = (1 - score) * 100
    return confidence

def get_confidence_info(confidence):
    if confidence >= 90:
        return "green", "🟢"  # Green circle for high confidence
    elif confidence >= 60:
        return "orange", "🟠"  # Orange circle for medium confidence
    else:
        return "red", "🔴"  # Red circle for low confidence

SYSTEM_PROMPT = """
Act strictly as an advanced AI-based transcription and notation tool, directly converting images of documents into detailed Markdown text. Start immediately with the transcription and relevant notations, such as the type of content and special features observed. Do not include any introductory sentences or summaries.

Specific guidelines:
1. **Figures and Diagrams:** Transcribe all details and explicitly state the nature of any diagrams or figures so that they can be reconstructed based on your notation.
2. **Titles and Captions:** Transcribe all text exactly as seen, labeling them as 'Title:' or 'Caption:'.
3. **Underlined, Highlighted, or Circled Items:** Transcribe all such items and explicitly identify them as 'Underlined:', 'Highlighted:', or 'Circled:' so that they can be reconstructed based on your notation.
4. **Charts and Graphs:** Transcribe all related data and clearly describe its type, like 'Bar chart:' or 'Line graph:' so that they can be reconstructed based on your notation.
5. **Organizational Charts:** Transcribe all details and specify 'Organizational chart:' so that they can be reconstructed based on your notation.
6. **Tables:** Transcribe tables exactly as seen and start with 'Table:' so that they can be reconstructed based on your notation.
7. **Annotations and Comments:** Transcribe all annotations and comments, specifying their nature, like 'Handwritten comment:' or 'Printed annotation:', so that they can be reconstructed based on your notation.
8. **General Image Content:** Describe all relevant images, logos, and visual elements, noting features like 'Hand-drawn logo:' or 'Computer-generated image:' so that they can be reconstructed based on your notation.
9. **Handwritten Notes:** Transcribe all and clearly label as 'Handwritten note:', specifying their location within the document and creating a unique ID for each one so that they can be reconstructed based on your notation.
10. **Page Layout:** Describe significant layout elements directly so that the document layout can be reconstructed.
11. **Redactions:** Note any redacted sections with 'Redacted area:' so that they can be identified and the visible context can be reconstructed.

Each transcription should be devoid of filler content, focusing solely on the precise documentation and categorization of the visible information.
"""

USER_PROMPT = """ 
Transcribe and categorize all visible information from the image precisely as it is presented. Ensure to include notations about content types, such as 'Handwritten note:' or 'Graph type:'. Begin immediately with the details, omitting any introductory language.
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
        max_tokens=MAX_TOKENS,
        temperature=0.1
    )
    return response.choices[0].message.content


def save_uploadedfile(uploadedfile):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

def process_pdf(file_path, page_progress_bar, page_status_text):
    doc = fitz.open(file_path)
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    page_contents = []

    total_pages = len(doc)
    for page_num in range(total_pages):
        page = doc[page_num]
        pix = page.get_pixmap()
        image_path = os.path.join(temp_dir, f"page{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append((page_num + 1, image_path))
        
        try:
            page_content = get_generated_data(image_path)
            page_contents.append(page_content)
        except Exception as e:
            st.error(f"Error processing page {page_num + 1}: {str(e)}")
            page_contents.append("")

        # Update progress
        progress = (page_num + 1) / total_pages
        page_progress_bar.progress(progress)
        page_status_text.text(f"Processing PDF page {page_num + 1} of {total_pages}")

    doc.close()
    return image_paths, page_contents

def docx_to_html(docx_path):
    with open(docx_path, "rb") as docx_file:
        result = mammoth.convert_to_html(docx_file, convert_image=mammoth.images.img_element)
        html = result.value
        messages = result.messages
        
        # Add minimal CSS to maintain document structure and style
        html = f"""
        <html>
        <head>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                }}
                p {{
                    margin-bottom: 1em;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    margin-top: 1em;
                    margin-bottom: 0.5em;
                }}
                table {{
                    border-collapse: collapse;
                    margin-bottom: 1em;
                }}
                td, th {{
                    border: 1px solid #ddd;
                    padding: 8px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
                @page {{
                    size: letter;
                    margin: 2cm;
                }}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """
        
        return html, messages


def crop_image(image_path):
    with Image.open(image_path) as img:
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        img.save(image_path)

def html_to_images(html_content, page_progress_bar, page_status_text):
    with Display():
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        
        # Use dynamic width and height calculation
        options = {
            'format': 'png',
            'quality': 100,
            'width': '0',  # Set a fixed width
            'height': '0'  # let height be determined by content
        }
        
        # Create a temporary HTML file for the entire document
        temp_html_path = os.path.join(temp_dir, "document.html")
        with open(temp_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Convert the HTML file to a PDF
        pdf_path = os.path.join(temp_dir, "document.pdf")
        try:
            pdfkit.from_file(temp_html_path, pdf_path, options=options)
        except Exception as e:
            st.error(f"Error converting HTML to PDF: {str(e)}")
            return []

        # Convert PDF to images
        image_paths = pdf_to_images(pdf_path, page_progress_bar, page_status_text)

    return image_paths


def html_to_pdf(html_content, output_pdf_path):
    pdfkit.from_string(html_content, output_pdf_path)

def pdf_to_images(pdf_path, page_progress_bar, page_status_text):
    doc = fitz.open(pdf_path)
    temp_dir = tempfile.mkdtemp()
    image_paths = []

    total_pages = len(doc)
    for page_num in range(total_pages):
        page = doc[page_num]
        pix = page.get_pixmap()
        image_path = os.path.join(temp_dir, f"page{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append((page_num + 1, image_path))

        # Update progress
        progress = (page_num + 1) / total_pages
        page_progress_bar.progress(progress)
        page_status_text.text(f"Converting PDF page {page_num + 1} of {total_pages} to image")

    doc.close()
    return image_paths

def process_doc_docx(file_path, page_progress_bar, page_status_text):
    try:
        # Convert DOCX to HTML
        page_status_text.text("Converting DOC/DOCX to HTML")
        html_content, messages = docx_to_html(file_path)
        
        # Log any messages from the conversion process
        for message in messages:
            if message.type == "warning":
                st.warning(f"Warning during conversion: {message.message}")
            elif message.type == "error":
                st.error(f"Error during conversion: {message.message}")
        
        # Convert HTML to images
        image_paths = html_to_images(html_content, page_progress_bar, page_status_text)
        
        page_contents = []

        # Process each image for AI text extraction
        total_pages = len(image_paths)
        for i, (page_num, image_path) in enumerate(image_paths):
            if image_path is None:
                page_contents.append("")  # Skip this page due to previous error
                continue
            
            try:
                page_content = get_generated_data(image_path)
                page_contents.append(page_content)
            except Exception as e:
                st.error(f"Error processing page {page_num}: {str(e)}")
                page_contents.append("")

            # Update progress
            progress = (i + 1) / total_pages
            page_progress_bar.progress(progress)
            page_status_text.text(f"Processing DOC/DOCX page {i + 1} of {total_pages}")

        return image_paths, page_contents
    except Exception as e:
        st.error(f"Error processing DOC/DOCX file: {str(e)}")
        return [], []


def process_txt(file_path, page_progress_bar, page_status_text):
    page_status_text.text("Processing TXT file")
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    page_progress_bar.progress(1.0)
    page_status_text.text("TXT file processing complete")
    return [(1, None)], [content]  # No image paths for TXT files

def process_excel(file_path, page_progress_bar, page_status_text):
    def dataframe_to_markdown(df):
        return df.to_markdown(index=False)

    excel_file = pd.ExcelFile(file_path)
    page_contents = []
    
    total_sheets = len(excel_file.sheet_names)
    for i, sheet_name in enumerate(excel_file.sheet_names):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        markdown_content = f"## Sheet: {sheet_name}\n\n{dataframe_to_markdown(df)}"
        page_contents.append(markdown_content)
        
        # Update progress
        progress = (i + 1) / total_sheets
        page_progress_bar.progress(progress)
        page_status_text.text(f"Processing Excel sheet {i + 1} of {total_sheets}")
    
    return [(i+1, None) for i in range(len(page_contents))], page_contents

def generate_summary(page_contents, progress_bar, status_text):
    total_tokens = sum(num_tokens_from_string(content) for content in page_contents)
    
    if total_tokens > MAX_TOKENS:
        # If the document is very large, we need to summarize it in chunks
        chunk_size = MAX_TOKENS // 2  # Leave room for the summary generation prompt
        chunks = []
        current_chunk = []
        current_chunk_tokens = 0
        
        for content in page_contents:
            content_tokens = num_tokens_from_string(content)
            if current_chunk_tokens + content_tokens > chunk_size:
                chunks.append("\n".join(current_chunk))
                current_chunk = [content]
                current_chunk_tokens = content_tokens
            else:
                current_chunk.append(content)
                current_chunk_tokens += content_tokens
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        summaries = []
        for i, chunk in enumerate(chunks):
            status_text.text(f"Summarizing chunk {i + 1} of {len(chunks)}")
            chunk_summary = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes document chunks."},
                    {"role": "user", "content": f"Provide a brief summary of this document chunk ({i+1}/{len(chunks)}):\n\n{chunk}"}
                ],
                max_tokens=MAX_TOKENS // 4  # Limit the summary size for each chunk
            ).choices[0].message.content
            summaries.append(chunk_summary)
            
            # Update progress
            progress = (i + 1) / len(chunks)
            progress_bar.progress(progress)
        
        status_text.text("Generating final summary")
        final_summary = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that combines document chunk summaries."},
                {"role": "user", "content": f"Combine these chunk summaries into a coherent overall summary:\n\n{''.join(summaries)}"}
            ],
            max_tokens=MAX_TOKENS // 2  # Limit the final summary size
        ).choices[0].message.content
    else:
        # If the document is not too large, we can summarize it in one go
        status_text.text("Generating summary")
        final_summary = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
                {"role": "user", "content": f"Provide a comprehensive summary of this document:\n\n{''.join(page_contents)}"}
            ],
            max_tokens=MAX_TOKENS // 2  # Limit the summary size
        ).choices[0].message.content
    
    progress_bar.progress(1.0)
    status_text.text("Summary generation complete")
    return final_summary

# Main processing function
def process_file(uploaded_file, overall_progress_bar, overall_status_text, file_index, total_files):
    file_progress_bar = st.progress(0)
    file_status_text = st.empty()
    page_progress_bar = st.progress(0)
    page_status_text = st.empty()

    st.subheader(f"Processing: {uploaded_file.name}")

    temp_file_path = save_uploadedfile(uploaded_file)
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    file_status_text.text("Initializing document processing...")
    file_progress_bar.progress(5)

    collection = get_or_create_collection("document_pages")
    if collection is None:
        st.error("Error in creating or accessing the collection.")
        return None, None, None, None

    image_paths = []
    page_contents = []

    try:
        file_status_text.text("Extracting content from file...")
        file_progress_bar.progress(10)

        if file_extension == '.pdf':
            image_paths, page_contents = process_pdf(temp_file_path, page_progress_bar, page_status_text)
        elif file_extension in ['.doc', '.docx']:
            image_paths, page_contents = process_doc_docx(temp_file_path, page_progress_bar, page_status_text)
        elif file_extension == '.txt':
            image_paths, page_contents = process_txt(temp_file_path, page_progress_bar, page_status_text)
        elif file_extension in ['.xls', '.xlsx']:
            image_paths, page_contents = process_excel(temp_file_path, page_progress_bar, page_status_text)
        elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
            image_paths = [(1, temp_file_path)]
            page_content = get_generated_data(temp_file_path)
            page_contents = [page_content]
            page_progress_bar.progress(100)
            page_status_text.text("Image processed")
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None, None, None, None

        if not page_contents:
            st.error(f"No content extracted from the file: {uploaded_file.name}")
            return None, None, None, None

        file_status_text.text("Generating summary...")
        file_progress_bar.progress(40)
        summary = generate_summary(page_contents, file_progress_bar, file_status_text)  # Fixed: added missing arguments
        
        file_status_text.text("Storing pages in vector database...")
        file_progress_bar.progress(60)
        
        total_pages = len(page_contents)
        for i, content in enumerate(page_contents):
            page_vector = embeddings.embed_documents([content])[0]
            entity = {
                "content": content,
                "file_name": uploaded_file.name,
                "page_number": i + 1,
                "vector": page_vector,
                "summary": summary
            }
            collection.insert([entity])
            progress_percentage = 60 + (i + 1) / total_pages * 35  # Progress from 60% to 95%
            file_progress_bar.progress(int(progress_percentage))
            file_status_text.text(f"Storing page {i+1} of {total_pages}...")
            page_progress_bar.progress(int((i + 1) / total_pages * 100))
            page_status_text.text(f"Storing page {i+1} of {total_pages}")

            # Update overall progress
            overall_progress = (file_index * 100 + progress_percentage) / total_files
            overall_progress_bar.progress(int(overall_progress))
            overall_status_text.text(f"Processing file {file_index + 1} of {total_files}: {uploaded_file.name}")

        file_progress_bar.progress(100)
        file_status_text.text("Processing complete!")
        page_progress_bar.progress(100)
        page_status_text.text("All pages processed!")

        # Store the processed data in session state
        st.session_state.documents[uploaded_file.name] = {
            'image_paths': image_paths,
            'page_contents': page_contents,
            'summary': summary
        }

        return collection, image_paths, page_contents, summary
    except Exception as e:
        st.error(f"An error occurred while processing {uploaded_file.name}: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        
        # Additional debugging information
        st.write(f"Debug: File extension: {file_extension}")
        st.write(f"Debug: Temp file path: {temp_file_path}")
        if os.path.exists(temp_file_path):
            st.write(f"Debug: Temp file size: {os.path.getsize(temp_file_path)} bytes")
        else:
            st.write("Debug: Temp file does not exist")
        
        return None, None, None, None

def search_documents(query, selected_documents):
    collection = get_or_create_collection("document_pages")
    if collection is None:
        st.error("Error in creating or accessing the collection.")
        return []

    collection.load()
    
    query_vector = embeddings.embed_query(query)
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }

    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=1000,
        expr=f"file_name in {selected_documents}",
        output_fields=["content", "file_name", "page_number", "summary"]
    )

    all_pages = []
    for hit in results[0]:
        confidence = calculate_confidence(hit.score)
        page = {
            'file_name': hit.entity.get('file_name'),
            'content': hit.entity.get('content'),
            'page_number': hit.entity.get('page_number'),
            'score': hit.score,
            'confidence': confidence,
            'summary': hit.entity.get('summary')
        }
        all_pages.append(page)

    return all_pages

# Streamlit interface
st.title('📄 Document Query App')

# Initialize session state variables
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'file_hashes' not in st.session_state:
    st.session_state.file_hashes = {}
if 'files_to_remove' not in st.session_state:
    st.session_state.files_to_remove = set()
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Advanced Options")
    citation_length = st.slider("Maximum length of each citation", 100, 1000, 250)
    
    if st.button("🗑️ Clear Current Session"):
        st.session_state.documents = {}
        st.session_state.file_hashes = {}
        st.session_state.files_to_remove = set()
        st.success("Current session cleared. You can still access previously uploaded documents.")
        st.rerun()

    st.markdown("## ℹ️ About")
    st.info(
        "This app allows you to upload PDF documents, Markdown files, or images, "
        "extract information from them, and query the content. "
        "It uses OpenAI's GPT-4 model for text generation and "
        "Milvus for efficient similarity search across sessions."
    )
    
    st.markdown("## 📖 How to use")
    st.info(
        "1. Upload one or more PDF, Markdown, or image files.\n"
        "2. Wait for the processing to complete.\n"
        "3. Select the documents you want to query (including from previous sessions).\n"
        "4. Enter your query in the text box.\n"
        "5. Click 'Search' to get answers based on the selected document content.\n"
        "6. View the answer and sources.\n"
        "7. Optionally, export the Q&A session as a PDF."
    )

    st.markdown("## ⚠️ Note")
    st.warning(
        "This is a prototype application. Do not upload sensitive "
        "information. In the deployed version, there will be a "
        "private database to ensure security and privacy."
    )

# Main app section for file upload and processing
# Main app
try:
    connect_to_milvus()

    # File upload section
    uploaded_files = st.file_uploader("📤 Upload PDF, Word, TXT, Excel, or Image file(s)", 
                                      type=["pdf", "doc", "docx", "txt", "xls", "xlsx", "png", "jpg", "jpeg", "tiff", "bmp", "gif"], 
                                      accept_multiple_files=True)
    if uploaded_files:
        overall_progress_bar = st.progress(0)
        overall_status_text = st.empty()
        
        for file_index, uploaded_file in enumerate(uploaded_files):
            file_content = uploaded_file.getvalue()
            file_hash = get_file_hash(file_content)
            
            if file_hash in st.session_state.file_hashes:
                existing_file_name = st.session_state.file_hashes[file_hash]
                st.success(f"File '{uploaded_file.name}' has already been processed as '{existing_file_name}'. Using existing data.")
                # Update overall progress
                overall_progress = ((file_index + 1) * 100) / len(uploaded_files)
                overall_progress_bar.progress(int(overall_progress))
                overall_status_text.text(f"Skipped file {file_index + 1} of {len(uploaded_files)}: {uploaded_file.name} (already processed)")
            else:
                try:
                    with st.spinner('Processing file... This may take a while for large documents.'):
                        collection, image_paths, page_contents, summary = process_file(uploaded_file, overall_progress_bar, overall_status_text, file_index, len(uploaded_files))
                    if collection is not None:
                        st.session_state.documents[uploaded_file.name] = {
                            'image_paths': image_paths,
                            'page_contents': page_contents,
                            'summary': summary
                        }
                        st.session_state.file_hashes[file_hash] = uploaded_file.name
                        st.success(f"File processed and stored in vector database!")
                        # with st.expander("📑 View Summary"):
                        #     st.markdown(f"🗂️ **Document Summary**\n\n{summary}")
                except Exception as e:
                    st.error(f"An error occurred while processing {uploaded_file.name}: {str(e)}")
        
        overall_progress_bar.progress(100)
        overall_status_text.text("All files processed!")

    # Document selection section
    st.divider()
    st.subheader("📂 All Available Documents")

    all_documents = list(set(get_all_documents() + list(st.session_state.documents.keys())))

    if all_documents:
        selected_documents = st.multiselect(
            "Select documents to view or query:",
            options=all_documents,
            default=list(st.session_state.documents.keys())
        )
    else:
        st.info("No documents available. Please upload some documents to get started.")
        selected_documents = []

    # Query interface and answer display
    st.divider()
    st.subheader("🔍 Query the Document(s)")
    query = st.text_input("Enter your query about the document(s):")
    search_button = st.button("🔎 Search")
                
    if search_button and selected_documents:
        with st.spinner('Searching...'):
            all_pages = search_documents(query, selected_documents)
            
            if not all_pages:
                st.warning("No relevant results found. Please try a different query.")
            else:
                content = "\n".join([f"[{page['file_name']}-p{page['page_number']}] {page['content']}" for page in all_pages])

                system_content = "You are an assisting agent. Please provide a detailed response based on the input. After your response, list the sources of information used, including file names, page numbers, and relevant snippets. Make full use of the available context to provide comprehensive answers. Include citation IDs in your response for easy verification."
                user_content = f"Respond to the query '{query}' using the information from the following content: {content}"

                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=MAX_TOKENS
                )
                st.divider()
                st.subheader("💬 Answer:")
                                
                # Process the response to add clickable links and confidence indicators
                answer_text = response.choices[0].message.content
                for page in all_pages:
                    citation_id = f"{page['file_name']}-p{page['page_number']}"
                    _, confidence_icon = get_confidence_info(page['confidence'])
                    replacement = f"[{citation_id}]({citation_id}){confidence_icon}"
                    answer_text = answer_text.replace(f"[{citation_id}]", replacement)

                st.markdown(answer_text)
                
                # Add JavaScript to scroll to the source when a citation is clicked
                st.markdown("""
                <script>
                const citations = document.querySelectorAll('a[href^="#"]');
                citations.forEach(citation => {
                    citation.addEventListener('click', function(e) {
                        e.preventDefault();
                        const targetId = this.getAttribute('href').slice(1);
                        const targetElement = document.getElementById(targetId);
                        if (targetElement) {
                            targetElement.scrollIntoView({behavior: 'smooth'});
                        }
                    });
                });
                </script>
                """, unsafe_allow_html=True)
                
                st.divider()
                st.subheader("📚 Sources:")
                                
                # Group sources by file
                sources_by_file = {}
                for page in all_pages:
                    sources_by_file.setdefault(page['file_name'], []).append(page)

                total_citation_length = 0
                for file_name, pages in sources_by_file.items():
                    st.markdown(f"### 📄 {file_name}")
                    for page in pages:
                        confidence = page['confidence']
                        color, icon = get_confidence_info(confidence)
                        
                        col1, col2 = st.columns([1, 9])
                        
                        with col1:
                            st.markdown(f"<span style='color:{color};'>●</span> {icon} **{confidence:.1f}%**", unsafe_allow_html=True)
                        
                        with col2:
                            citation_id = f"{file_name}-p{page['page_number']}"
                            st.markdown(f"<div id='{citation_id}'></div>", unsafe_allow_html=True)
                            st.markdown(f"**Page {page['page_number']}**")
                            
                            content_to_display = page['content'][:citation_length]
                            full_content = page['content']
                            
                            st.markdown(f"[{citation_id}] {content_to_display}" + ("..." if len(page['content']) > citation_length else ""))
                            
                            if len(page['content']) > citation_length:
                                with st.expander("Show Full Content"):
                                    st.markdown(full_content)
                            
                            total_citation_length += len(content_to_display)
                        
                        if file_name in st.session_state.documents:
                            image_paths = st.session_state.documents[file_name]['image_paths']
                            image_path = next((img_path for num, img_path in image_paths if num == page['page_number']), None)
                            if image_path:
                                with st.expander("Show Image"):
                                    st.image(image_path, use_column_width=True, caption=f"Page {page['page_number']}")
                        
                        st.markdown("---")

                with st.expander("📊 Document Statistics", expanded=False):
                    st.write(f"Total pages searched: {len(all_pages)}")
                    st.write(f"Total citation length: {total_citation_length} characters")
                    for page in all_pages:
                        st.write(f"File: {page['file_name']}, Page: {page['page_number']}, Confidence: {page['confidence']:.2f}%")

                # Save question and answer to history
                st.session_state.qa_history.append({
                    'question': query,
                    'answer': response.choices[0].message.content,
                    'sources': [{'file': page['file_name'], 'page': page['page_number'], 'confidence': page['confidence']} for page in all_pages],
                    'documents_queried': selected_documents
                })


                
    elif search_button:
        st.warning("Please select at least one document to query.")

    # Document content display
    if selected_documents:
        st.divider()
        st.subheader("**📜 Document Content:**")
        for file_name in selected_documents:
            st.subheader(f"📄 {file_name}")
            page_contents = get_document_content(file_name)
            if page_contents:
                with st.expander("🗂️ Document Summary", expanded=True):
                    st.markdown(page_contents[0]['summary'])
                
                for page in page_contents:
                    with st.expander(f"📑Page {page['page_number']}"):
                        st.markdown(page['content'])
                        
                        if file_name in st.session_state.documents:
                            image_paths = st.session_state.documents[file_name]['image_paths']
                            image_path = next((img_path for num, img_path in image_paths if num == page['page_number']), None)
                            if image_path:
                                try:
                                    st.image(image_path, use_column_width=True, caption=f"Page {page['page_number']}")
                                except Exception as e:
                                    st.error(f"Error displaying image for page {page['page_number']}: {str(e)}")
                            else:
                                st.info(f"No image available for page {page['page_number']}")
            else:
                st.info(f"No content available for {file_name}.")
            
            if st.button(f"🗑️ Remove {file_name}", key=f"remove_{file_name}"):
                st.session_state.files_to_remove.add(file_name)
                st.rerun()

    # Remove files marked for deletion
    if st.session_state.files_to_remove:
        for file_name in list(st.session_state.files_to_remove):
            collection = get_or_create_collection("document_pages")
            collection.delete(f"file_name == '{file_name}'")
            if file_name in st.session_state.documents:
                del st.session_state.documents[file_name]
            st.success(f"{file_name} has been removed.")
        st.session_state.files_to_remove.clear()
        st.rerun()

    # Display question history
    if st.session_state.qa_history:
        st.divider()
        st.subheader("📜 Question History")
        for i, qa in enumerate(reversed(st.session_state.qa_history)):
            with st.expander(f"Q{len(st.session_state.qa_history)-i}: {qa['question']}"):
                st.write(f"A: {qa['answer']}")
                st.write("Documents Queried:", ", ".join(qa['documents_queried']))
                st.write("Sources:")
                for source in qa['sources']:
                    st.write(f"- File: {source['file']}, Page: {source['page']}")

        if st.button("🗑️ Clear Question History"):
            st.session_state.qa_history = []
            st.success("Question history cleared!")
            st.rerun()

        # Export results
        if st.button("📤 Export Q&A Session"):
            qa_session = ""
            for qa in st.session_state.qa_history:
                qa_session += f"Q: {qa['question']}\n\nA: {qa['answer']}\n\nDocuments Queried: {', '.join(qa['documents_queried'])}\n\nSources:\n"
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

# Warning banner and terms (place this at the end of your script, outside the main try-except block)
st.markdown("""
<div class="warning-banner">
    <span class="big-font">⚠️ IMPORTANT NOTICE</span><br>
    This is a prototype application. Do not upload sensitive information as it is accessible to anyone. 
    In the deployed version, there will be a private database to ensure security and privacy.
</div>
""", unsafe_allow_html=True)

with st.expander("⚠️ By using this application, you agree to the following terms and conditions:", expanded=True):
    st.markdown("""
    <div class="bottom-warning">
        <ol style="text-align: left;">
            <li><strong>Multi-User Environment:</strong> Any data you upload or queries you make may be accessible to other users.</li>
            <li><strong>No Privacy:</strong> Do not upload any sensitive or confidential information.</li>
            <li><strong>Data Storage:</strong> All uploaded data is stored in Milvus and is not secure.</li>
            <li><strong>Accuracy:</strong> AI models may produce inaccurate or inconsistent results. Verify important information.</li>
            <li><strong>Liability:</strong> Use this application at your own risk. We are not liable for any damages or losses.</li>
            <li><strong>Data Usage:</strong> Uploaded data may be used to improve the application. We do not sell or intentionally share your data with third parties.</li>
            <li><strong>User Responsibilities:</strong> You are responsible for the content you upload and queries you make. Do not use this application for any illegal or unauthorized purpose.</li>
            <li><strong>Changes to Terms:</strong> We reserve the right to modify these terms at any time.</li>
        </ol>
        By continuing to use this application, you acknowledge that you have read, understood, and agree to these terms.
    </div>
    """, unsafe_allow_html=True)