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
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType, MilvusClient, MilvusException
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
from bs4 import BeautifulSoup
import traceback
import time
from streamlit.runtime.scriptrunner import RerunException
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx



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

def connect_to_milvus():
    connections.connect(
        alias="default",
        uri=MILVUS_ENDPOINT,
        token=MILVUS_API_KEY,
        secure=True
    )
def reset_session():
    # Clear session state variables
    st.session_state.clear()

    # Create a placeholder for the success message
    message = st.empty()
    message.success("Session reset successfully!")

    # Delay to allow UI to update
    time.sleep(2)

    # Clear the success message
    message.empty()

    # Reload the app using JavaScript
    st.write("""
        <script>
            window.location.reload();
        </script>
    """, unsafe_allow_html=True)

def get_or_create_custom_query_collection():
    collection_name = "custom_queries"
    try:
        if utility.has_collection(collection_name):
            return Collection(collection_name)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="query_part", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536)
            ]
            schema = CollectionSchema(fields, "Custom query collection")
            collection = Collection(collection_name, schema)
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index("vector", index_params)
            return collection
    except Exception as e:
        st.error(f"Error in creating or accessing the AI task collection: {str(e)}")
        return None

def get_or_create_document_pages_collection():
    collection_name = "document_pages"
    try:
        if utility.has_collection(collection_name):
            return Collection(collection_name)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
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
        st.error(f"Error in creating or accessing the document pages collection: {str(e)}")
        return None

def get_all_custom_queries():
    collection = get_or_create_custom_query_collection()
    if collection is None:
        return []

    try:
        collection.load()
        results = collection.query(
            expr="name != ''",
            output_fields=["name", "query_part"],
            limit=1000
        )
        return results
    except Exception as e:
        st.error(f"Error fetching custom AI task: {str(e)}")
        return []
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

#custom buttons

def save_custom_query(name, query_part, update=False):
    collection = get_or_create_custom_query_collection()
    if collection is None:
        return False

    try:
        # Generate embeddings for the query part
        embedding = embeddings.embed_documents([query_part])[0]

        if update:
            # Delete existing query
            collection.delete(expr=f"name == '{name}'")

        # Insert new or updated query
        collection.insert([{"name": name, "query_part": query_part, "vector": embedding}])
        return True
    except Exception as e:
        st.error(f"Error saving custom AI task: {str(e)}")
        return False

def update_custom_query(name, new_query_part):
    collection = get_or_create_custom_query_collection()
    if collection is None:
        return False

    try:
        # Generate embeddings for the new query part
        embedding = embeddings.embed_documents([new_query_part])[0]

        # Check if the query exists
        existing_query = collection.query(
            expr=f"name == '{name}'",
            output_fields=["id"],
            limit=1
        )

        if existing_query:
            # Delete the existing entry
            collection.delete(expr=f"name == '{name}'")

            # Insert the updated entry
            collection.insert([{
                "name": name,
                "query_part": new_query_part,
                "vector": embedding
            }])
        else:
            # If it doesn't exist, insert as new
            collection.insert([{
                "name": name,
                "query_part": new_query_part,
                "vector": embedding
            }])

        return True
    except Exception as e:
        st.error(f"Error updating custom AI task: {str(e)}")
        return False


def use_custom_query(query_name, query, selected_documents):
    custom_queries = get_all_custom_queries()
    st.write(f"Debug: Custom AI task fetched: {custom_queries}")
    for custom_query in custom_queries:
        if custom_query['name'] == query_name:
            query_part = custom_query['query_part']
            full_query = f"{query_part} {query}"
            st.write(f"Debug: Full AI task formed: {full_query}")
            return search_documents(full_query, selected_documents, custom_prompt=query_part)
    return [], None

def delete_custom_query(name):
    collection = get_or_create_custom_query_collection()
    if collection is None:
        st.error("Failed to access custom query collection")
        return

    try:
        collection.delete(f"name == '{name}'")
        st.session_state.custom_queries = get_all_custom_queries()
        # Delay to allow UI to update
        time.sleep(1)
        # Use the reset_session function to reset the session state and rerun the app
        reset_session()
    except Exception as e:
        st.error(f"Error deleting custom AI task: {str(e)}")

def remove_document(file_name):
    collection = get_or_create_document_pages_collection()
    if collection is None:
        st.error("Failed to access document pages collection")
        return False

    try:
        # Delete document from Milvus collection
        delete_result = collection.delete(expr=f"file_name == '{file_name}'")

        # Display the delete result
        st.write(f"Delete result: {delete_result}")

        # Check if any documents were deleted
        if delete_result.delete_count > 0:
            # Remove document from session state
            st.session_state.documents.pop(file_name, None)
            if 'selected_documents' in st.session_state and file_name in st.session_state.selected_documents:
                st.session_state.selected_documents.remove(file_name)

            # Set a flag to indicate successful removal
            st.session_state.document_removed = True
            st.session_state.removed_document_name = file_name

            st.success(f"Document '{file_name}' successfully removed from the collection.")
            return True
        else:
            st.warning(f"No document '{file_name}' found in the collection to delete.")
            return False
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return False

#sources
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

# Initialize session state variables if they don't exist
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'file_hashes' not in st.session_state:
    st.session_state.file_hashes = {}
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'custom_queries' not in st.session_state:
    st.session_state.custom_queries = get_all_custom_queries()
if 'custom_query_selected' not in st.session_state:
    st.session_state.custom_query_selected = False
if 'query_part_clicked' not in st.session_state:
    st.session_state.query_part_clicked = None
if 'query_name_clicked' not in st.session_state:
    st.session_state.query_name_clicked = None
if 'files_to_remove' not in st.session_state:
    st.session_state.files_to_remove = []
if 'selected_documents' not in st.session_state:
    st.session_state.selected_documents = []
if st.session_state.get('document_removed', False):
    st.success(f"{st.session_state.removed_document_name} has been removed.")
    st.session_state.document_removed = False
    st.session_state.removed_document_name = None

def display_results(all_pages, custom_response, query, selected_documents):
    content = "\n".join([f"[{page['file_name']}-p{page['page_number']}] {page['content']}" for page in all_pages])

    if custom_response is None:
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
        answer_text = response.choices[0].message.content
    else:
        answer_text = custom_response

    st.divider()
    st.subheader("💬 Answer:")

    # Process the response to add clickable links and confidence indicators
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
                    with st.expander("📑Show Full Content"):
                        st.markdown(full_content)

                total_citation_length += len(content_to_display)

            if file_name in st.session_state.documents:
                image_paths = st.session_state.documents[file_name]['image_paths']
                image_path = next((img_path for num, img_path in image_paths if num == page['page_number']), None)
                if image_path:
                    with st.expander("🖼️Show Image"):
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
        'answer': answer_text,
        'sources': [{'file': page['file_name'], 'page': page['page_number'], 'confidence': page['confidence']} for page in all_pages],
        'documents_queried': selected_documents
    })


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

# Add a new function to process CSV files
def process_csv(file_path, page_progress_bar, page_status_text):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Convert DataFrame to markdown
    markdown_content = df.to_markdown(index=False)

    # Split the content into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=60000,  # Adjust this value as needed, keeping it under 65535
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(markdown_content)

    page_contents = chunks

    # Update progress
    page_progress_bar.progress(1.0)
    page_status_text.text(f"CSV file processed into {len(chunks)} chunks")

    # Return a list of tuples (page_number, None) for each chunk, and the page contents
    return [(i+1, None) for i in range(len(chunks))], page_contents

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

def docx_to_html_with_breaks(docx_path):
    # Custom style map to preserve explicit page breaks
    style_map = """
    p[style-name='Page Break'] => div.explicit-page-break
    """

    # Convert DOCX to HTML using mammoth
    with open(docx_path, "rb") as docx_file:
        result = mammoth.convert_to_html(docx_file, style_map=style_map)
        html = result.value

    # Get page setup information using python-docx
    doc = Document(docx_path)
    section = doc.sections[0]
    page_height = section.page_height.cm
    page_width = section.page_width.cm
    top_margin = section.top_margin.cm
    bottom_margin = section.bottom_margin.cm
    left_margin = section.left_margin.cm
    right_margin = section.right_margin.cm

    # Calculate available content area
    content_height = page_height - top_margin - bottom_margin
    content_width = page_width - left_margin - right_margin

    # Parse HTML and insert page breaks based on content volume
    soup = BeautifulSoup(html, 'html.parser')

    # Ensure we have an html tag
    if not soup.html:
        new_html = soup.new_tag('html')
        new_html.extend(soup.contents)
        soup = BeautifulSoup(str(new_html), 'html.parser')

    # Ensure we have a head tag
    if not soup.head:
        head = soup.new_tag('head')
        soup.html.insert(0, head)

    # Ensure we have a body tag
    if not soup.body:
        body = soup.new_tag('body')
        for tag in list(soup.html.children):
            if tag.name != 'head':
                body.append(tag.extract())
        soup.html.append(body)

    current_height = 0
    for tag in soup.find_all(['p', 'div', 'table']):
        if tag.name == 'div' and 'explicit-page-break' in tag.get('class', []):
            current_height = 0
            continue

        # Estimate element height (this is a simplified calculation)
        if tag.name == 'p':
            element_height = 1  # Assume 1 cm for each paragraph
        elif tag.name == 'table':
            rows = len(tag.find_all('tr'))
            element_height = rows * 0.5  # Assume 0.5 cm for each table row
        else:
            element_height = 0.5  # Default height for other elements

        if current_height + element_height > content_height:
            new_break = soup.new_tag('div', **{'class': 'content-based-break'})
            tag.insert_before(new_break)
            current_height = element_height
        else:
            current_height += element_height

    # Add CSS for proper rendering
    style = soup.new_tag('style')
    style.string = """
        body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
        p { margin: 0; padding: 0; }
        .explicit-page-break, .content-based-break { page-break-after: always; height: 0; }
    """
    soup.head.append(style)

    return str(soup)

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

        # Use minimal options to avoid width/height issues
        options = {
            'format': 'png',
            'quality': '100',
            'disable-smart-width': '',
            'log-level': 'none'  # Suppress non-error messages
        }

        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all page break divs
        page_breaks = soup.find_all('div', class_=['explicit-page-break', 'content-based-break'])

        # Split the content into pages
        pages = []
        current_page = []
        for element in soup.body.children:
            if element in page_breaks:
                if current_page:
                    pages.append(''.join(str(tag) for tag in current_page))
                current_page = []
            else:
                current_page.append(element)
        if current_page:
            pages.append(''.join(str(tag) for tag in current_page))

        total_pages = len(pages)

        for i, page_content in enumerate(pages):
            # Create a temporary HTML file for each page
            temp_html_path = os.path.join(temp_dir, f"page_{i+1}.html")
            with open(temp_html_path, 'w', encoding='utf-8') as f:
                f.write(f"""
                <html>
                    <head>
                        <style>
                            body {{
                                width: 210mm;
                                height: 297mm;
                                margin: 0;
                                padding: 0;
                            }}
                            * {{
                                box-sizing: border-box;
                            }}
                        </style>
                    </head>
                    <body>{page_content}</body>
                </html>
                """)

            # Convert the HTML file to an image
            image_path = os.path.join(temp_dir, f"page{i + 1}.png")
            try:
                imgkit.from_file(temp_html_path, image_path, options=options)
                crop_image(image_path)
                image_paths.append((i + 1, image_path))
            except Exception as e:
                st.error(f"Error processing page {i + 1}: {str(e)}")
                image_paths.append((i + 1, None))  # Record the error but continue processing

            # Update progress
            progress = (i + 1) / total_pages
            page_progress_bar.progress(progress)
            page_status_text.text(f"Converting page {i + 1} of {total_pages} to image")

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
        # Step 1: Convert DOCX to HTML
        page_status_text.text("Converting DOC/DOCX to HTML")
        html_content = docx_to_html_with_breaks(file_path)

        if not html_content:
            raise ValueError("HTML content is empty after conversion")

        # Step 2: Convert HTML to images
        try:
            image_paths = html_to_images(html_content, page_progress_bar, page_status_text)
        except Exception as e:
            st.error(f"Error during HTML to image conversion: {str(e)}")
            st.error(f"Full error: {traceback.format_exc()}")
            return [], []

        if not image_paths:
            raise ValueError("No images were generated from the HTML content")

        # Step 3: Process images and extract content
        page_contents = []
        total_pages = len(image_paths)
        for i, (page_num, image_path) in enumerate(image_paths):
            if image_path is None:
                st.warning(f"Skipping page {page_num} due to previous error")
                page_contents.append("")
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

        if not page_contents:
            raise ValueError("No content was extracted from the document")

        return image_paths, page_contents
    except Exception as e:
        st.error(f"Error processing DOC/DOCX file: {str(e)}")
        st.error(f"Full error: {traceback.format_exc()}")
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

#session states



def handle_new_query(name, query_part):
    if save_custom_query(name, query_part):
        st.session_state.custom_queries = get_all_custom_queries()
        message = st.empty()
        message.success(f"Custom query '{name}' saved successfully!")
        time.sleep(2)
        message.empty()
        st.rerun()
    else:
        st.error(f"Failed to save custom query '{name}'")

def verify_collection_exists(collection_name):
    if utility.has_collection(collection_name):
        st.success(f"Collection '{collection_name}' exists.")
        return True
    else:
        st.warning(f"Collection '{collection_name}' does not exist.")
        return False
def reset_custom_query_states():
    st.session_state.custom_query_selected = False
    st.session_state.query_part_clicked = None
    st.session_state.query_name_clicked = None
    st.session_state.custom_queries = get_all_custom_queries()

def handle_update_query(name, new_query_part):
    if save_custom_query(name, new_query_part, update=True):
        st.session_state.custom_queries = get_all_custom_queries()
        st.success(f"Updated {name}")

def handle_delete_query(name):
    if delete_custom_query(name):
        st.session_state.custom_queries = get_all_custom_queries()
        st.success(f"Deleted {name}")

def handle_remove_document(file_name):
    if remove_document(file_name):
        # Update the list of available documents
        st.session_state.documents = get_all_documents()
        st.success(f"Removed document: {file_name}")
        # Set a flag to trigger a rerun
        st.session_state.document_removed = True
        st.session_state.removed_document_name = file_name
    else:
        st.error(f"Failed to remove document: {file_name}")
def remove_question(index):
    if 0 <= index < len(st.session_state.qa_history):
        del st.session_state.qa_history[index]
        message = st.empty()
        message.success("Question removed successfully")
        time.sleep(2)
        message.empty()
        st.rerun()
        return True
    st.error("Invalid question index")
    return False

def clear_current_session():
    st.session_state.documents = {}
    st.session_state.file_hashes = {}
    st.session_state.qa_history = []
    if 'selected_documents' in st.session_state:
        st.session_state.selected_documents = []
    st.session_state.custom_queries = get_all_custom_queries()
    message = st.empty()
    message.success("Current session cleared")
    time.sleep(2)
    message.empty()
    st.rerun()


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

        if file_extension == '.csv':
            image_paths, page_contents = process_csv(temp_file_path, page_progress_bar, page_status_text)
        elif file_extension == '.pdf':
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
        summary = generate_summary(page_contents, file_progress_bar, file_status_text)

        file_status_text.text("Storing pages in vector database...")
        file_progress_bar.progress(60)

        total_pages = len(page_contents)
        for i, content in enumerate(page_contents):
            page_number = i + 1
            # Check if the page already exists in the collection
            query_result = collection.query(
                expr=f"file_name == '{uploaded_file.name}' and page_number == {page_number}",
                output_fields=["id"],
                limit=1
            )
            if query_result:
                st.info(f"Page {page_number} of '{uploaded_file.name}' already exists in Milvus.")
            else:
                page_vector = embeddings.embed_documents([content])[0]
                entity = {
                    "content": content[:65000],  # Truncate content to fit within Milvus VARCHAR limit
                    "file_name": uploaded_file.name,
                    "page_number": page_number,
                    "vector": page_vector,
                    "summary": summary[:65000] if summary else ""  # Truncate summary as well
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
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None, None


def search_documents(query, selected_documents, custom_prompt=None):
    collection = get_or_create_collection("document_pages")
    if collection is None:
        st.error("Error in creating or accessing the collection.")
        return [], None

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

    if custom_prompt:
        content = "\n".join([f"[{page['file_name']}-p{page['page_number']}] {page['content']}" for page in all_pages])
        system_content = "You are an assisting agent. Please provide a detailed response based on the input. After your response, list the sources of information used, including file names, page numbers, and relevant snippets. Make full use of the available context to provide comprehensive answers. Include citation IDs in your response for easy verification."
        user_content = custom_prompt.format(query=query, content=content)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            max_tokens=MAX_TOKENS
        )
        return all_pages, response.choices[0].message.content
    else:
        return all_pages, None

# Streamlit interface
st.title('💡 DocuQuery AI')


# Sidebar
with st.sidebar:
    st.header("⚙️ Advanced Options")
    citation_length = st.slider("Maximum length of each citation", 100, 1000, 250)

    if st.button("🗑️ Clear Current Session"):
        clear_current_session()
        st.success("Current session cleared. You can still access previously uploaded documents.")
        st.rerun()

    # Move Custom Query Macros section here
    st.divider()
    st.subheader("📌 Custom AI Tasks")
    custom_queries = get_all_custom_queries()

    if custom_queries:
        for custom_query in custom_queries:
            if st.button(f"📌 {custom_query['name']}", key=f"custom_query_{custom_query['name']}"):
                st.session_state.query_part_clicked = custom_query['query_part']
                st.session_state.query_name_clicked = custom_query['name']
                st.session_state.custom_query_selected = True

    # Add new custom query
    with st.expander("➕ Add New Custom AI Task"):
        with st.form(key="new_custom_query_form"):
            new_query_name = st.text_input("AI Task Name")
            new_query_part = st.text_area("AI Task Instructions")
            submit_button = st.form_submit_button("Save Custom Query")
            if submit_button:
                handle_new_query(new_query_name, new_query_part)

    # Edit or delete existing custom queries
    with st.expander("✏️ Edit or Delete Custom AI Tasks"):
        for index, query in enumerate(st.session_state.custom_queries):
            with st.form(key=f"edit_custom_query_form_{index}"):
                st.markdown(f"### {query['name']}")
                edited_query_part = st.text_area(f"AI task for {query['name']}", query['query_part'])
                col1, col2 = st.columns(2)
                with col1:
                    update_button = st.form_submit_button(f"Update {query['name']}")
                with col2:
                    delete_button = st.form_submit_button(f"Delete {query['name']}")

                if update_button:
                    handle_update_query(query['name'], edited_query_part)
                elif delete_button:
                    delete_custom_query(query['name'])

    st.markdown("## ℹ️ About")
    st.info(
        "This app allows you to upload PDF documents, Markdown files, or images, "
        "extract information from them, and query the content. "
        "It uses OpenAI's GPT-4 model for text generation and "
        "Milvus for efficient similarity search across sessions."
    )

    st.markdown("## 📖 How to use")
    st.info(
        "1. Upload one or more files (PDF, Word, TXT, Excel, or images).\n"
        "2. Wait for the AI to process and analyze the documents.\n"
        "3. Select the documents you want to analyze (including from previous sessions).\n"
        "4. Describe your task or question in the text box. Be specific for best results.\n"
        "5. Click 'Execute AI Task' to  insights based on the selected documents.\n"
        "6. Review the AI's response and explore the cited sources.\n"
        "7. Use custom AI tasks for quick, predefined analyses.\n"
        "8. Optionally, export your analysis session as a PDF for future reference."
    )

    st.markdown("### 🛠️ Managing Custom AI Tasks")
    st.info(
        "Custom AI tasks are predefined analyses you can run quickly on your documents. Here's how to manage them:\n\n"
        "**Adding a new custom task:**\n"
        "1. Expand the 'Add New Custom Task' section in the sidebar.\n"
        "2. Enter a name for your task and describe what the AI should do.\n"
        "3. Click 'Save Custom Task' to add it to your list.\n\n"
        "**Using a custom task:**\n"
        "1. Find your task in the 'Custom AI Tasks' list in the sidebar.\n"
        "2. Click on the task to apply it to your selected documents.\n\n"
        "**Editing a custom task:**\n"
        "1. Expand the 'Edit or Delete Custom Tasks' section in the sidebar.\n"
        "2. Find the task you want to edit and modify its description.\n"
        "3. Click 'Update' to save your changes.\n\n"
        "**Removing a custom task:**\n"
        "1. In the 'Edit or Delete Custom Tasks' section, find the task you want to remove.\n"
        "2. Click 'Delete' next to the task to remove it from your list."
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
    # Your main logic goes here
    connect_to_milvus()

    # File upload section
    uploaded_files = st.file_uploader("📤 Upload PDF, Word, TXT, Excel, CSV, or Image file(s)", 
                                    type=["pdf", "doc", "docx", "txt", "xls", "xlsx", "csv", "png", "jpg", "jpeg", "tiff", "bmp", "gif"], 
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
    st.subheader("🤖 AI-Powered Document Task")
    query = st.text_input("Enter your task or question about the document(s):")
    search_button = st.button("🚀 Execute AI Task")

    if 'custom_query_selected' not in st.session_state:
        st.session_state.custom_query_selected = False

    # Check if documents are selected before processing any query
    if not selected_documents:
        if search_button or st.session_state.custom_query_selected:
            st.warning("Please select at least one document to query.")
    else:
        if search_button or st.session_state.custom_query_selected:
            spinner_message = "Applying custom query and searching..." if st.session_state.custom_query_selected else "Searching documents..."

            with st.spinner(spinner_message):
                if st.session_state.custom_query_selected:
                    full_query = f"{st.session_state.query_part_clicked} {query}"
                    display_query = f"{st.session_state.query_name_clicked}: {full_query}"
                    st.session_state.custom_query_selected = False
                else:
                    full_query = query
                    display_query = query

                all_pages, custom_response = search_documents(full_query, selected_documents)

                if not all_pages:
                    st.warning("No relevant results found in the selected documents.")
                else:
                    display_results(all_pages, custom_response, display_query, selected_documents)

    # Reset the custom query selection flag
    st.session_state.custom_query_selected = False

    # Document content display
    if selected_documents:
        st.divider()
        st.subheader("**📜 Document Content:**")
        for file_name in selected_documents:
            st.subheader(f"📄 {file_name}")
            page_contents = get_document_content(file_name)
            if page_contents:
                with st.expander("🗂️ Document Summary", expanded=False):
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
            
            # Place the remove button here, after displaying the document content
            if st.button(f"🗑️ Remove {file_name}", key=f"remove_{file_name}"):
                if remove_document(file_name):
                    st.rerun()
                else:
                    st.error(f"Failed to remove {file_name}. Please check the logs for more information.")

    # Display question history
    if st.session_state.qa_history:
        st.divider()
        st.subheader("📜 Task History")

        for i, qa in enumerate(reversed(st.session_state.qa_history)):
            question_index = len(st.session_state.qa_history) - i - 1
            with st.expander(f"Q{i+1}: {qa['question']}"):
                st.markdown(f"**Answer:**\n{qa['answer']}")
                st.markdown(f"**Documents Queried:** {', '.join(qa['documents_queried'])}")
                st.markdown("**Sources:**")
                for source in qa['sources']:
                    st.markdown(f"- File: {source['file']}, Page: {source['page']}")

                if st.button(f"🗑️ Remove this Q&A", key=f"remove_qa_{question_index}"):
                    if remove_question(question_index):
                        st.success("Question and answer removed.")
                    else:
                        st.error("Failed to remove question and answer. Please try again.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All History"):
                st.session_state.qa_history = []
                st.success("All question history cleared!")
        with col2:
            if st.button("📥 Download History as CSV"):
                csv = "Question,Answer,Documents Queried,Sources\n"
                for qa in st.session_state.qa_history:
                    question = qa['question'].replace('"', '""')
                    answer = qa['answer'].replace('"', '""')
                    documents = "|".join(qa['documents_queried']).replace('"', '""')
                    sources = "|".join([f"{s['file']} (Page {s['page']})" for s in qa['sources']]).replace('"', '""')
                    csv += f'"{question}","{answer}","{documents}","{sources}"\n'

                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="qa_history.csv",
                    mime="text/csv"
                )
    else:
        st.info("No question history available.")

    # Export results
    if st.button("📤 Export Session"):
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
# States

# At the end of your main script
if st.session_state.get('document_removed', False):
    st.success(f"{st.session_state.removed_document_name} has been removed.")
    st.session_state.document_removed = False
    st.session_state.removed_document_name = None
    st.rerun()