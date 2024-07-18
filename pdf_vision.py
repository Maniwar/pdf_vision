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

# User Agreement
USER_AGREEMENT = """
# User Agreement for Document Query and Analysis App

By using this application, you agree to the following terms and conditions:

1. Data Storage and Security:
   - All data uploaded to this application, including documents and queries, will be stored in a database.
   - While we strive to maintain security, there is always a risk of data breaches. By using this application, you acknowledge and accept this risk.
   - DO NOT upload any confidential, sensitive, or personal information to this application.

2. Data Usage and Access:
   - The data you upload may be used to improve the application's functionality and performance.
   - We do not sell or share your data with third parties.
   - However, please be aware that other users of this application may potentially interact with or access data you have uploaded. Do not assume any uploaded information is private or secure.

3. Accuracy of Results:
   - This application uses advanced language models to process and analyze documents.
   - These models can sometimes produce inaccurate or inconsistent results.
   - Do not rely solely on the information provided by this application for critical decisions.
   - Always verify important information through other sources.

4. User Responsibilities:
   - You are responsible for ensuring that you have the right to upload and process any documents you submit to the application.
   - You agree not to use this application for any illegal or unauthorized purpose.
   - You agree not to attempt to access or manipulate data uploaded by other users.

5. Limitation of Liability:
   - The creators and operators of this application are not liable for any damages or losses resulting from your use of the application, including but not limited to data loss, exposure, or misuse by other users.

6. No Expectation of Privacy:
   - Given the nature of this application, you should have no expectation of privacy for any information you upload or input into the system.
   - Treat all interactions with this application as potentially public.

7. Changes to the Agreement:
   - We reserve the right to modify this agreement at any time. Continued use of the application after changes constitutes acceptance of the new terms.

By using this application, you acknowledge that you have read, understood, and agree to these terms and conditions.
"""

# Set the API key using st.secrets for secure access
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini"
client = OpenAI()
embeddings = OpenAIEmbeddings()

# CSS for Warning Banner
st.markdown("""
<style>
    .warning-banner {
        background-color: #ffcccb;
        border: 1px solid #ff0000;
        padding: 10px;
        color: #a00;
        font-weight: bold;
        text-align: center;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# HTML for Warning Banner
st.markdown("""
<div class="warning-banner">
    Warning: This is a prototype application. Do not upload sensitive information as it is accessible to anyone. In the deployed version, there will be a private database to ensure security and privacy.
</div>
""", unsafe_allow_html=True)

def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

SYSTEM_PROMPT = "You are a helpful assistant that responds in Markdown. Help me with Given Image Extraction with Given Details with Different categories!"
USER_PROMPT = """
Retrieve all the information provided in the image, including figures, titles, highlighted items, circled words, charts, and graphs, as well as all the values from graphs and charts and or relationships between entities, so users can ask questions about these items as needed.
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
        max_tokens=1000,
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
st.title('Document Query and Analysis App')

st.warning("""
    ⚠️ IMPORTANT: This is a multi-user environment. Any data you upload or queries you make
    may be accessible to other users. Do not upload sensitive or confidential information.
    Use this application at your own risk.
""")

# User Agreement
st.markdown("---")
with st.expander("User Agreement"):
    st.markdown(USER_AGREEMENT)

# Checkbox for user to accept the agreement
if st.checkbox("I have read and agree to the User Agreement"):
    st.success("Thank you for accepting the User Agreement. You may now use the application.")
else:
    st.warning("You must accept the User Agreement to use this application.")
    st.stop()  # This will prevent the rest of the app from running if the agreement is not accepted

if st.checkbox("I have read and agree to the User Agreement"):
    st.success("Thank you for accepting the User Agreement. You may now use the application.")
    
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
                    
                    st.subheader("Answer:")
                    st.write(response.choices[0].message.content)

                    confidence_score = calculate_confidence(all_docs)
                    st.write(f"Confidence Score: {confidence_score}%")

                    st.subheader("Sources:")
                    for file_name, doc, score in all_docs:
                        page_num = doc.metadata.get('page_number', 'Unknown')
                        st.markdown(f"**File: {file_name}, Page {page_num}, Relevance: {1 - score:.2f}**")
                        highlighted_text = highlight_relevant_text(doc.page_content[:200], query)
                        st.markdown(f"```\n{highlighted_text}...\n```")
                        
                        image_path = next((img_path for num, img_path in st.session_state['processed_data'][file_name]['image_paths'] if num == page_num), None)
                        if image_path:
                            with st.expander(f"View Page {page_num} Image"):
                                st.image(image_path, use_column_width=True)

                    st.write(f"Debug - Total documents retrieved: {len(all_docs)}")
                    for file_name, doc, score in all_docs:
                        st.write(f"Debug - File: {file_name}, Page: {doc.metadata.get('page_number', 'Unknown')}, Score: {1 - score:.2f}")
                        st.write(f"Debug - Content snippet: {doc.page_content[:50]}...")

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

else:
    st.warning("You must accept the User Agreement to use this application.")
    st.stop()  # This will prevent the rest of the app from running if the agreement is not accepted
if __name__ == "__main__":
    st.sidebar.markdown("## About")
    st.sidebar.info(
        "This app allows you to upload PDF documents or images, "
        "extract information from them, and query the content. "
        "It uses OpenAI's GPT model for text generation and "
        "FAISS for efficient similarity search."
    )
    
    st.sidebar.markdown("## How to use")
    st.sidebar.info(
        "1. Upload one or more PDF or image files.\n"
        "2. Wait for the processing to complete.\n"
        "3. Enter your query in the text box.\n"
        "4. Click 'Search' to get answers based on the document content.\n"
        "5. View the answer, confidence score, and sources.\n"
        "6. Optionally, export the Q&A session as a PDF."
    )

    st.sidebar.markdown("## Note")
    st.sidebar.warning(
        "This is a prototype application. Do not upload sensitive "
        "information. In the deployed version, there will be a "
        "private database to ensure security and privacy."
    )
