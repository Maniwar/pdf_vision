import os
import base64
import streamlit as st
from pathlib import Path
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import fitz  # PyMuPDF for handling PDFs
import tempfile
import markdown2
import pdfkit
from PIL import Image
import hashlib

# Set page configuration to wide mode
st.set_page_config(layout="wide")

# Set the API key using st.secrets for secure access
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini"
client = OpenAI()
embeddings = OpenAIEmbeddings()

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

def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
SYSTEM_PROMPT = """
Act strictly as an advanced AI based transcription and notation tool, directly converting images of documents into detailed Markdown text. Start immediately with the transcription and relevant notations, such as the type of content and special features observed. Do not include any introductory sentences or summaries.

Specific guidelines:
1. **Figures and Diagrams:** Transcribe all details and explicitly state the nature of any diagrams or figures.
2. **Titles and Captions:** Transcribe all text exactly as seen, labelling them as 'Title:' or 'Caption:'.
3. **Highlighted or Circled Items:** Transcribe all and explicitly mark them as 'Highlighted text:' or 'Circled text:'.
4. **Charts and Graphs:** Transcribe all and clearly describe its type, like 'Bar chart:', 'Line graph:'.
5. **Organizational Charts:** Transcribe all details and specify 'Organizational chart:'.
6. **Tables:** Transcribe all exactly as seen and start with 'Table:'.
7. **Annotations and Comments:** Transcribe all annotations and comments, specifying their nature, like 'Handwritten comment:' or 'Printed annotation:'.
8. **General Image Content:** Describe all relevant images, logos, and visual elements, noting features like 'Hand-drawn logo:', 'Computer-generated image:'.
9. **Handwritten Notes:** Transcribe all and clearly label as 'Handwritten note:', also note the location where each hand written notes can be found in the document. Create a unique ID for each one.
10. **Page Layout:** Describe significant layout elements directly.
11. **Redactions:** Note any redacted sections with 'Redacted area:'.

Each transcription should be concise and devoid of filler content, focusing solely on the precise documentation and categorization of the visible information.
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

def generate_summary(content):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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

def split_markdown_by_pages(markdown_content):
    pages = markdown_content.split('\n## Page ')
    return [page.strip() for page in pages if page.strip()]

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

    pages = split_markdown_by_pages(markdown_content)
    data = []
    for i, page_content in enumerate(pages):
        data.append({
            'content': page_content,
            'metadata': {'page_number': i + 1}
        })

    vector_db = FAISS.from_texts(
        [item['content'] for item in data],
        embeddings,
        metadatas=[item['metadata'] for item in data]
    )

    summary = generate_summary(markdown_content)

    progress_bar.progress(100)

    return vector_db, image_paths, markdown_content, summary

# Streamlit interface
st.title('📄 Document Query and Analysis App')

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
        st.header("⚙️ Advanced Options")
        chunks_to_retrieve = st.slider("Number of chunks to retrieve", 1, 10, 5)
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5)

        if st.button("🗑️ Clear Current Session"):
            st.session_state['current_session_files'] = set()
            st.session_state['file_hashes'] = {}
            st.success("Current session cleared. You can now upload new files.")

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
                        st.success(f"File processed and stored in vector database! Summary: {summary}")
                except Exception as e:
                    st.error(f"An error occurred while processing {uploaded_file.name}: {str(e)}")

            # Display summary and extracted content
            display_name = uploaded_file.name if uploaded_file.name in st.session_state['processed_data'] else st.session_state['file_hashes'].get(file_hash, uploaded_file.name)
            with st.expander(f"📑 View Summary for {display_name}"):
                st.markdown(st.session_state['processed_data'][display_name]['summary'])
            with st.expander(f"📄 View Extracted Content for {display_name}"):
                st.markdown(st.session_state['processed_data'][display_name]['markdown_content'])

    # Display all uploaded images for the current session
    if st.session_state['current_session_files']:
        st.subheader("📁 Uploaded Documents and Images")
        for file_name in st.session_state['current_session_files']:
            with st.expander(f"🖼️ Images from {file_name}"):
                for page_num, image_path in st.session_state['processed_data'][file_name]['image_paths']:
                    st.image(image_path, caption=f"Page {page_num}", use_column_width=True)

    # Query interface
    st.subheader("🔍 Query the Document(s)")
    query = st.text_input("Enter your query about the document(s):")
    if st.button("🔎 Search"):
        if st.session_state['current_session_files']:
            with st.spinner('Searching...'):
                all_docs = []
                for file_name in st.session_state['current_session_files']:
                    vector_db = st.session_state['processed_data'][file_name]['vector_db']
                    docs = vector_db.similarity_search_with_score(query, k=chunks_to_retrieve)
                    all_docs.extend([(file_name, doc, score) for doc, score in docs])
                
                # Sort all_docs by relevance score
                all_docs.sort(key=lambda x: x[2])
                
                content = "\n".join([f"File: {file_name}, Page {doc.metadata.get('page_number', 'Unknown')}: {doc.page_content}" for file_name, doc, _ in all_docs])

                system_content = "You are an assisting agent. Please provide the response based on the input. After your response, list the sources of information used, including file names, page numbers, and relevant snippets."
                user_content = f"Respond to the query '{query}' using the information from the following content: {content}"

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ]
                )
                
                st.subheader("💬 Answer:")
                st.write(response.choices[0].message.content)

                confidence_score = calculate_confidence(all_docs)
                st.write(f"Confidence Score: {confidence_score}%")

                st.subheader("📚 Sources:")
                for file_name, doc, score in all_docs:
                    page_num = doc.metadata.get('page_number', 'Unknown')
                    st.markdown(f"**File: {file_name}, Page {page_num}, Relevance: {1 - score:.2f}**")
                    highlighted_text = highlight_relevant_text(doc.page_content[:200], query)
                    st.markdown(f"```\n{highlighted_text}...\n```")
                    
                    image_path = next((img_path for num, img_path in st.session_state['processed_data'][file_name]['image_paths'] if num == page_num), None)
                    if image_path:
                        with st.expander(f"🖼️ View Page {page_num} Image"):
                            st.image(image_path, use_column_width=True)

                st.write(f"Total documents retrieved: {len(all_docs)}")
                for file_name, doc, score in all_docs:
                    st.write(f"File: {file_name}, Page: {doc.metadata.get('page_number', 'Unknown')}, Score: {1 - score:.2f}")
                    st.write(f"Content snippet: {doc.page_content[:100]}...")

            # Save question and answer to history
            if 'qa_history' not in st.session_state:
                st.session_state['qa_history'] = []
            st.session_state['qa_history'].append({
                'question': query,
                'answer': response.choices[0].message.content,
                'sources': [{'file': file_name, 'page': doc.metadata.get('page_number', 'Unknown')} for file_name, doc, _ in all_docs],
                'confidence': confidence_score
            })

        else:
            st.warning("Please upload and process at least one file first.")

    # Display question history
    if 'qa_history' in st.session_state and st.session_state['qa_history']:
        st.subheader("📜 Question History")
        for i, qa in enumerate(st.session_state['qa_history']):
            with st.expander(f"Q{i+1}: {qa['question']}"):
                st.write(f"A: {qa['answer']}")
                st.write(f"Confidence: {qa['confidence']}%")
                st.write("Sources:")
                for source in qa['sources']:
                    st.write(f"- File: {source['file']}, Page: {source['page']}")
        
        # Add a button to clear the question history
        if st.button("🗑️ Clear Question History"):
            st.session_state['qa_history'] = []
            st.success("Question history cleared!")

    # Export results
    if st.button("📤 Export Q&A Session"):
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

if __name__ == "__main__":
    st.sidebar.markdown("## ℹ️ About")
    st.sidebar.info(
        "This app allows you to upload PDF documents or images, "
        "extract information from them, and query the content. "
        "It uses OpenAI's GPT model for text generation and "
        "FAISS for efficient similarity search."
    )
    
    st.sidebar.markdown("## 📖 How to use")
    st.sidebar.info(
        "1. Upload one or more PDF or image files.\n"
        "2. Wait for the processing to complete.\n"
        "3. Enter your query in the text box.\n"
        "4. Click 'Search' to get answers based on the document content.\n"
        "5. View the answer, confidence score, and sources.\n"
        "6. Optionally, export the Q&A session as a PDF."
    )

    st.sidebar.markdown("## ⚠️ Note")
    st.sidebar.warning(
        "This is a prototype application. Do not upload sensitive "
        "information. In the deployed version, there will be a "
        "private database to ensure security and privacy."
    )

    st.markdown("""
    <div class="big-font">⚠️ IMPORTANT NOTICE</div>
    This is a prototype application. By using this application, you agree to the following terms and conditions:

    1. **Multi-User Environment**: Any data you upload or queries you make may be accessible to other users.
    2. **No Privacy**: Do not upload any sensitive or confidential information.
    3. **Data Storage**: All uploaded data is stored temporarily and is not secure.
    4. **Accuracy**: AI models may produce inaccurate or inconsistent results. Verify important information.
    5. **Liability**: Use this application at your own risk. We are not liable for any damages or losses.
    6. **Data Usage**: Uploaded data may be used to improve the application. We do not sell or intentionally share your data with third parties.
    7. **User Responsibilities**: You are responsible for the content you upload and queries you make. Do not use this application for any illegal or unauthorized purpose.
    8. **Changes to Terms**: We reserve the right to modify these terms at any time.

    By continuing to use this application, you acknowledge that you have read, understood, and agree to these terms.
    """, unsafe_allow_html=True)
