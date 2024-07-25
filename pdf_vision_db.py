import os
import base64
import streamlit as st
from pathlib import Path
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF for handling PDFs
import tempfile
import markdown2
import pdfkit
import hashlib
import tiktoken
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

# Set page configuration to wide mode
st.set_page_config(layout="wide")

# Set the API key using st.secrets for secure access
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini"
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

# Warning Banner
st.markdown("""
<div class="warning-banner">
    <span class="big-font">⚠️ IMPORTANT NOTICE</span><br>
    This is a prototype application. Do not upload sensitive information as it is accessible to anyone. 
    In the deployed version, there will be a private database to ensure security and privacy.
</div>
""", unsafe_allow_html=True)

def connect_to_milvus():
    connections.connect(
        alias="default", 
        uri=MILVUS_ENDPOINT,
        token=MILVUS_API_KEY,
        secure=True
    )

def get_or_create_collection(collection_name, dim=1536):  # 1536 is the dimension for OpenAI embeddings
    if utility.has_collection(collection_name):
        return Collection(collection_name)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields, "Document chunks collection")
        collection = Collection(collection_name, schema)
        
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index("vector", index_params)
        return collection

def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_content(content: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=num_tokens_from_string,
    )
    chunks = text_splitter.split_text(content)
    return chunks

def get_all_documents():
    collection = get_or_create_collection("document_chunks")
    collection.load()
    results = collection.query(
        expr="file_name != ''",
        output_fields=["file_name"],
        limit=10000  # Adjust this limit as needed
    )
    return list(set(doc['file_name'] for doc in results))

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
        max_tokens=16384,
        temperature=0.1
    )
    return response.choices[0].message.content

def save_uploadedfile(uploadedfile):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

def generate_summary(chunks):
    summaries = []
    for i, chunk in enumerate(chunks):
        chunk_summary = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes document chunks."},
                {"role": "user", "content": f"Provide a brief summary of this document chunk ({i+1}/{len(chunks)}):\n\n{chunk}"}
            ]
        ).choices[0].message.content
        summaries.append(chunk_summary)
    
    final_summary = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that combines document chunk summaries."},
            {"role": "user", "content": f"Combine these chunk summaries into a coherent overall summary:\n\n{''.join(summaries)}"}
        ]
    ).choices[0].message.content
    
    return final_summary

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

    chunks = chunk_content(markdown_content)
    
    collection = get_or_create_collection("document_chunks")
    collection.load()

    chunk_vectors = embeddings.embed_documents(chunks)
    entities = [
        chunks,
        [uploaded_file.name] * len(chunks),
        list(range(len(chunks))),
        chunk_vectors
    ]
    collection.insert(entities)

    summary = generate_summary(chunks)

    progress_bar.progress(100)

    return collection, image_paths, chunks, summary

# Streamlit interface
st.title('📄 Document Query and Analysis App')

try:
    connect_to_milvus()

    # Initialize session state variables
    if 'current_session_files' not in st.session_state:
        st.session_state['current_session_files'] = set()
    if 'processed_data' not in st.session_state:
        st.session_state['processed_data'] = {}
    if 'file_hashes' not in st.session_state:
        st.session_state['file_hashes'] = {}

    # Load all existing files from Milvus
    all_documents = get_all_documents()

    # Sidebar for advanced options
    with st.sidebar:
        st.header("⚙️ Advanced Options")
        chunks_to_retrieve = st.slider("Number of chunks to retrieve", 1, 10, 5)
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5)

        if st.button("🗑️ Clear Current Session"):
            st.session_state['current_session_files'] = set()
            st.session_state['processed_data'] = {}
            st.session_state['file_hashes'] = {}
            st.success("Current session cleared. You can still access previously uploaded documents.")

    # File upload section
    uploaded_files = st.file_uploader("📤 Upload PDF or Image file(s)", type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif"], accept_multiple_files=True)
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
                    with st.spinner('Processing file... This may take a while for large documents.'):
                        collection, image_paths, chunks, summary = process_file(uploaded_file)
                    if collection is not None:
                        st.session_state['processed_data'][uploaded_file.name] = {
                            'image_paths': image_paths,
                            'chunks': chunks,
                            'summary': summary
                        }
                        st.session_state['current_session_files'].add(uploaded_file.name)
                        st.session_state['file_hashes'][file_hash] = uploaded_file.name
                        all_documents.append(uploaded_file.name)
                        st.success(f"File processed and stored in vector database!")
                        with st.expander("View Summary"):
                            st.markdown(summary)
                except Exception as e:
                    st.error(f"An error occurred while processing {uploaded_file.name}: {str(e)}")

    # Document Selection and Management
    st.divider()
    st.subheader("📂 All Available Documents")

    all_documents = list(set(all_documents + list(st.session_state['current_session_files'])))

    if all_documents:
        selected_documents = st.multiselect(
            "Select documents to query:",
            options=all_documents,
            default=list(st.session_state['current_session_files'])
        )
        
        for file_name in selected_documents:
            with st.expander(f"📄 {file_name}"):
                if file_name in st.session_state['processed_data']:
                    st.markdown(f"**Summary:**")
                    st.markdown(st.session_state['processed_data'][file_name]['summary'])
                    
                    st.markdown("**Images:**")
                    for page_num, image_path in st.session_state['processed_data'][file_name]['image_paths']:
                        st.image(image_path, caption=f"Page {page_num}", use_column_width=True)
                else:
                    st.info(f"Detailed information for {file_name} is not available in the current session. You can still query this document.")
                
                if st.button(f"🗑️ Remove {file_name}", key=f"remove_{file_name}"):
                    collection = get_or_create_collection("document_chunks")
                    collection.delete(f"file_name == '{file_name}'")
                    all_documents.remove(file_name)
                    if file_name in st.session_state['current_session_files']:
                        st.session_state['current_session_files'].remove(file_name)
                    if file_name in st.session_state['processed_data']:
                        del st.session_state['processed_data'][file_name]
                    st.success(f"{file_name} has been removed.")
                    st.rerun()
    else:
        st.info("No documents available. Please upload some documents to get started.")

    # Query interface
    st.divider()
    st.subheader("🔍 Query the Document(s)")
    query = st.text_input("Enter your query about the document(s):")
    if st.button("🔎 Search"):
        if selected_documents:
            with st.spinner('Searching...'):
                collection = get_or_create_collection("document_chunks")
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
                    limit=chunks_to_retrieve,
                    expr=f"file_name in {selected_documents}",
                    output_fields=["content", "file_name", "chunk_index"]
                )

                all_docs = []
                for hit in results[0]:
                    all_docs.append((
                        hit.entity.get('file_name'),
                        {
                            'content': hit.entity.get('content'),
                            'chunk_index': hit.entity.get('chunk_index')
                        },
                        hit.distance
                    ))
                
                # Sort all_docs by relevance score
                all_docs.sort(key=lambda x: x[2])
                
                content = "\n".join([f"File: {file_name}, Chunk {doc['chunk_index']}: {doc['content']}" for file_name, doc, _ in all_docs])

                system_content = "You are an assisting agent. Please provide the response based on the input. After your response, list the sources of information used, including file names, chunk indices, and relevant snippets."
                user_content = f"Respond to the query '{query}' using the information from the following content: {content}"

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ]
                )
                st.divider()
                st.subheader("💬 Answer:")
                st.write(response.choices[0].message.content)

                confidence_score = calculate_confidence(all_docs)
                st.write(f"Confidence Score: {confidence_score}%")

                st.divider()
                st.subheader("📚 Sources:")
                for file_name, doc, score in all_docs:
                    chunk_index = doc['chunk_index']
                    st.markdown(f"**File: {file_name}, Chunk {chunk_index}, Relevance: {1 - score:.2f}**")
                    highlighted_text = highlight_relevant_text(doc['content'][:200], query)
                    st.markdown(f"```\n{highlighted_text}...\n```")
                    
                    # Find the corresponding image
                    if file_name in st.session_state['processed_data']:
                        image_paths = st.session_state['processed_data'][file_name]['image_paths']
                        page_num = chunk_index // 2 + 1  # Assuming 2 chunks per page, adjust as needed
                        image_path = next((img_path for num, img_path in image_paths if num == page_num), None)
                        if image_path:
                            with st.expander(f"🖼️ View Image: {file_name}, Page {page_num}"):
                                st.image(image_path, use_column_width=True)

                with st.expander("📊 Document Statistics", expanded=False):
                    st.write(f"Total chunks retrieved: {len(all_docs)}")
                    for file_name, doc, score in all_docs:
                        st.write(f"File: {file_name}, Chunk: {doc['chunk_index']}, Score: {1 - score:.2f}")
                        st.write(f"Content snippet: {doc['content'][:100]}...")

                # Save question and answer to history
                if 'qa_history' not in st.session_state:
                    st.session_state['qa_history'] = []
                st.session_state['qa_history'].append({
                    'question': query,
                    'answer': response.choices[0].message.content,
                    'sources': [{'file': file_name, 'chunk': doc['chunk_index']} for file_name, doc, _ in all_docs],
                    'confidence': confidence_score,
                    'documents_queried': selected_documents
                })

        else:
            st.warning("Please select at least one document to query.")
    
    # Display question history
    if 'qa_history' in st.session_state and st.session_state['qa_history']:
        st.divider()
        st.subheader("📜 Question History")
        for i, qa in enumerate(st.session_state['qa_history']):
            with st.expander(f"Q{i+1}: {qa['question']}"):
                st.write(f"A: {qa['answer']}")
                st.write(f"Confidence: {qa['confidence']}%")
                st.write("Documents Queried:", ", ".join(qa['documents_queried']))
                st.write("Sources:")
                for source in qa['sources']:
                    st.write(f"- File: {source['file']}, Chunk: {source['chunk']}")
        
        # Add a button to clear the question history
        if st.button("🗑️ Clear Question History"):
            st.session_state['qa_history'] = []
            st.success("Question history cleared!")

        # Export results
        if st.button("📤 Export Q&A Session"):
            qa_session = ""
            for qa in st.session_state.get('qa_history', []):
                qa_session += f"Q: {qa['question']}\n\nA: {qa['answer']}\n\nConfidence: {qa['confidence']}%\n\nDocuments Queried: {', '.join(qa['documents_queried'])}\n\nSources:\n"
                for source in qa['sources']:
                    qa_session += f"- File: {source['file']}, Chunk: {source['chunk']}\n"
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

if __name__ == "__main__":
    st.sidebar.markdown("## ℹ️ About")
    st.sidebar.info(
        "This app allows you to upload PDF documents or images, "
        "extract information from them, and query the content. "
        "It uses OpenAI's GPT model for text generation and "
        "Milvus for efficient similarity search across sessions."
    )
    
    st.sidebar.markdown("## 📖 How to use")
    st.sidebar.info(
        "1. Upload one or more PDF or image files.\n"
        "2. Wait for the processing to complete.\n"
        "3. Select the documents you want to query (including from previous sessions).\n"
        "4. Enter your query in the text box.\n"
        "5. Click 'Search' to get answers based on the selected document content.\n"
        "6. View the answer, confidence score, and sources.\n"
        "7. Optionally, export the Q&A session as a PDF."
    )

    st.sidebar.markdown("## ⚠️ Note")
    st.sidebar.warning(
        "This is a prototype application. Do not upload sensitive "
        "information. In the deployed version, there will be a "
        "private database to ensure security and privacy."
    )

# Bottom warning section with expander
st.divider()
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
