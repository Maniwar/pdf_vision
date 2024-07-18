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
        if 'page_number' not in page.metadata:
            page.metadata['page_number'] = i + 1

    vector_db = Milvus.from_documents(
        data,
        embeddings,
        connection_args=MILVUS_CONNECTION_ARGS,
    )

    summary = generate_summary(markdown_content)

    progress_bar.progress(100)

    return vector_db, image_paths, markdown_content, summary

# Streamlit interface
st.title('Document Query and Analysis App')

try:
    # Sidebar for advanced options
    with st.sidebar:
        st.header("Advanced Options")
        chunks_to_retrieve = st.slider("Number of chunks to retrieve", 1, 10, 5)
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5)

    uploaded_files = st.file_uploader("Upload PDF or Image file(s)", type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if 'processed_data' not in st.session_state:
                st.session_state['processed_data'] = {}
            
            if uploaded_file.name not in st.session_state['processed_data']:
                try:
                    vector_db, image_paths, markdown_content, summary = process_file(uploaded_file)
                    if vector_db is not None:
                        st.session_state['processed_data'][uploaded_file.name] = {
                            'vector_db': vector_db,
                            'image_paths': image_paths,
                            'markdown_content': markdown_content,
                            'summary': summary
                        }
                        st.success(f"File processed and stored in vector database! Summary: {summary}")
                except Exception as e:
                    st.error(f"An error occurred while processing {uploaded_file.name}: {str(e)}")
                    st.exception(e)
            else:
                st.success(f"Using previously processed data for {uploaded_file.name}")

            with st.expander(f"View Summary for {uploaded_file.name}"):
                st.markdown(st.session_state['processed_data'][uploaded_file.name]['summary'])

            with st.expander(f"View Extracted Content for {uploaded_file.name}"):
                st.markdown(st.session_state['processed_data'][uploaded_file.name]['markdown_content'])

    # Query interface
    st.subheader("Query the Document(s)")
    query = st.text_input("Enter your query about the document(s):")
    if st.button("Search"):
        if 'processed_data' in st.session_state and st.session_state['processed_data']:
            with st.spinner('Searching...'):
                all_docs = []
                for file_name, data in st.session_state['processed_data'].items():
                    vector_db = data['vector_db']
                    docs = vector_db.similarity_search(query, k=chunks_to_retrieve)
                    all_docs.extend([(file_name, doc) for doc in docs])
                
                # Sort all_docs by relevance (assuming the order returned by similarity_search is from most to least relevant)
                all_docs.sort(key=lambda x: x[1].metadata.get('relevance', 0), reverse=True)
                
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

            # Save question and answer to history
            if 'qa_history' not in st.session_state:
                st.session_state['qa_history'] = []
            st.session_state['qa_history'].append({
                'question': query,
                'answer': response.choices[0].message.content,
                'sources': [{'file': file_name, 'page': doc.metadata.get('page_number', 'Unknown')} for file_name, doc in all_docs],
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
