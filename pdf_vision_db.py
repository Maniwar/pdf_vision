import os
import base64
import streamlit as st
from pathlib import Path
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
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
MODEL = "gpt-4o-mini"  # Latest GPT-4 model
client = OpenAI()
embeddings = OpenAIEmbeddings()

# Milvus connection parameters
MILVUS_ENDPOINT = st.secrets["general"]["MILVUS_PUBLIC_ENDPOINT"]
MILVUS_API_KEY = st.secrets["general"]["MILVUS_API_KEY"]

# iOS-like CSS styling (unchanged)
st.markdown("""
<style>
    /* iOS-like styling (unchanged) */
</style>
""", unsafe_allow_html=True)

# Warning Banner (unchanged)
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

def get_or_create_collection(collection_name, dim=1536):
    if utility.has_collection(collection_name):
        return Collection(collection_name)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
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
    collection = get_or_create_collection("document_pages")
    collection.load()
    results = collection.query(
        expr="file_name != ''",
        output_fields=["file_name"],
        limit=100000
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
        max_tokens=16000,
        temperature=0.1
    )
    return response.choices[0].message.content

def save_uploadedfile(uploadedfile):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

def generate_summary(page_contents):
    total_tokens = sum(num_tokens_from_string(content) for content in page_contents)
    max_tokens = 16000  # Adjust based on your model's context window
    
    if total_tokens > max_tokens:
        # If the document is very large, we need to summarize it in chunks
        chunk_size = max_tokens // 2  # Leave room for the summary generation prompt
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
            chunk_summary = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes document chunks."},
                    {"role": "user", "content": f"Provide a brief summary of this document chunk ({i+1}/{len(chunks)}):\n\n{chunk}"}
                ]
            ).choices[0].message.content
            summaries.append(chunk_summary)
        
        final_summary = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that combines document chunk summaries."},
                {"role": "user", "content": f"Combine these chunk summaries into a coherent overall summary:\n\n{''.join(summaries)}"}
            ]
        ).choices[0].message.content
    else:
        # If the document is not too large, we can summarize it in one go
        final_summary = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
                {"role": "user", "content": f"Provide a comprehensive summary of this document:\n\n{''.join(page_contents)}"}
            ]
        ).choices[0].message.content
    
    return final_summary

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
        page_contents = []
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            output = os.path.join(output_dir, f"page{page_num + 1}.png")
            pix.save(output)
            image_paths.append((page_num + 1, output))
            
            # Generate content for each page using GPT vision
            page_content = get_generated_data(output)
            page_contents.append(page_content)
            
            progress_bar.progress((page_num + 1) / total_pages)

        doc.close()
        st.success('PDF processed successfully!')
    elif file_extension == '.md':
        # Process Markdown using UnstructuredMarkdownLoader
        loader = UnstructuredMarkdownLoader(temp_file_path)
        data = loader.load()
        page_contents = [item.page_content for item in data]
        image_paths = []  # No images for Markdown files
        progress_bar.progress(1.0)
        st.success('Markdown processed successfully!')
    elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
        # Process single image
        image_paths = [(1, temp_file_path)]
        page_contents = [get_generated_data(temp_file_path)]
        progress_bar.progress(1.0)
        st.success('Image processed successfully!')
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return None, None, None, None

    # Use RecursiveCharacterTextSplitter to split the content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_text("\n".join(page_contents))

    collection = get_or_create_collection("document_pages")
    collection.load()

    page_vectors = embeddings.embed_documents(splits)
    entities = []
    for i, (split, vector) in enumerate(zip(splits, page_vectors)):
        entities.append({
            "content": split,
            "file_name": uploaded_file.name,
            "page_number": i + 1,  # This is now the chunk number rather than the page number
            "vector": vector
        })
    collection.insert(entities)

    summary = generate_summary(splits)

    progress_bar.progress(100)

    return collection, image_paths, splits, summary

def search_documents(query, selected_documents):
    collection = get_or_create_collection("document_pages")
    collection.load()
    
    query_vector = embeddings.embed_query(query)
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }

    # Initialize variables for pagination
    offset = 0
    limit = 16000  # Milvus max limit
    all_pages = []

    while True:
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=limit,
            offset=offset,
            expr=f"file_name in {selected_documents}",
            output_fields=["content", "file_name", "page_number"]
        )

        if not results[0]:  # If no results, break the loop
            break

        for hit in results[0]:
            page = {
                'file_name': hit.entity.get('file_name'),
                'content': hit.entity.get('content'),
                'page_number': hit.entity.get('page_number'),
                'score': hit.score
            }
            all_pages.append(page)

        offset += limit
        if len(results[0]) < limit:  # If we've retrieved all results, break the loop
            break

    return all_pages

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
        citations_to_display = st.slider("Number of citations to display", 1, 20, 5)
        
        if st.button("🗑️ Clear Current Session"):
            st.session_state['current_session_files'] = set()
            st.session_state['processed_data'] = {}
            st.session_state['file_hashes'] = {}
            st.success("Current session cleared. You can still access previously uploaded documents.")

    # File upload section
    uploaded_files = st.file_uploader("📤 Upload PDF, Markdown, or Image file(s)", type=["pdf", "md", "png", "jpg", "jpeg", "tiff", "bmp", "gif"], accept_multiple_files=True)
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
                        collection, image_paths, splits, summary = process_file(uploaded_file)
                    if collection is not None:
                        st.session_state['processed_data'][uploaded_file.name] = {
                            'image_paths': image_paths,
                            'splits': splits,
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
            st.subheader(f"📄 {file_name}")
            if file_name in st.session_state['processed_data']:
                st.markdown(f"**Summary:**")
                st.markdown(st.session_state['processed_data'][file_name]['summary'])
                
                st.markdown("**Content:**")
                for i, split in enumerate(st.session_state['processed_data'][file_name]['splits']):
                    with st.expander(f"Chunk {i+1}"):
                        st.markdown(split)
                        if st.session_state['processed_data'][file_name]['image_paths']:
                            image_path = next((img_path for num, img_path in st.session_state['processed_data'][file_name]['image_paths'] if num == i+1), None)
                            if image_path:
                                st.image(image_path, use_column_width=True)
            else:
                st.info(f"Detailed information for {file_name} is not available in the current session. You can still query this document.")
            
            if st.button(f"🗑️ Remove {file_name}", key=f"remove_{file_name}"):
                collection = get_or_create_collection("document_pages")
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
                all_pages = search_documents(query, selected_documents)
                
                if not all_pages:
                    st.warning("No relevant results found. Please try a different query.")
                else:
                    content = "\n".join([f"File: {page['file_name']}, Chunk: {page['page_number']}: {page['content']}" for page in all_pages])

                    system_content = "You are an assisting agent. Please provide the response based on the input. After your response, list the sources of information used, including file names, chunk numbers, and relevant snippets."
                    user_content = f"Respond to the query '{query}' using the information from the following content: {content}"

                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content}
                        ]
                    )
                    st.divider()
                    st.subheader("💬 Answer:")
                    st.write(response.choices[0].message.content)

                    st.divider()
                    st.subheader("📚 Sources:")
                    for i, page in enumerate(all_pages[:citations_to_display]):
                        st.markdown(f"**Source {i+1}: File: {page['file_name']}, Chunk: {page['page_number']}, Relevance: {1 - page['score']:.2f}**")
                        st.markdown(page['content'])
                        
                        if page['file_name'] in st.session_state['processed_data']:
                            image_paths = st.session_state['processed_data'][page['file_name']]['image_paths']
                            image_path = next((img_path for num, img_path in image_paths if num == page['page_number']), None)
                            if image_path:
                                st.image(image_path, use_column_width=True)
                        st.divider()

                    with st.expander("📊 Document Statistics", expanded=False):
                        st.write(f"Total chunks searched: {len(all_pages)}")
                        st.write(f"Citations displayed: {min(citations_to_display, len(all_pages))}")
                        for page in all_pages[:citations_to_display]:
                            st.write(f"File: {page['file_name']}, Chunk: {page['page_number']}, Score: {1 - page['score']:.2f}")

                    # Save question and answer to history
                    if 'qa_history' not in st.session_state:
                        st.session_state['qa_history'] = []
                    st.session_state['qa_history'].append({
                        'question': query,
                        'answer': response.choices[0].message.content,
                        'sources': [{'file': page['file_name'], 'chunk': page['page_number']} for page in all_pages[:citations_to_display]],
                        'documents_queried': selected_documents
                    })

        else:
            st.warning("Please select at least one document to query.")

    # Display question history
    if 'qa_history' in st.session_state and st.session_state['qa_history']:
        st.divider()
        st.subheader("📜 Question History")
        for i, qa in enumerate(reversed(st.session_state['qa_history'])):
            with st.expander(f"Q{len(st.session_state['qa_history'])-i}: {qa['question']}"):
                st.write(f"A: {qa['answer']}")
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
                qa_session += f"Q: {qa['question']}\n\nA: {qa['answer']}\n\nDocuments Queried: {', '.join(qa['documents_queried'])}\n\nSources:\n"
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
        "This app allows you to upload PDF documents, Markdown files, or images, "
        "extract information from them, and query the content. "
        "It uses OpenAI's GPT model for text generation and "
        "Milvus for efficient similarity search across sessions."
    )
    
    st.sidebar.markdown("## 📖 How to use")
    st.sidebar.info(
        "1. Upload one or more PDF, Markdown, or image files.\n"
        "2. Wait for the processing to complete.\n"
        "3. Select the documents you want to query (including from previous sessions).\n"
        "4. Enter your query in the text box.\n"
        "5. Click 'Search' to get answers based on the selected document content.\n"
        "6. View the answer and sources.\n"
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
