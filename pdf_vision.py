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

# Streamlit interface
st.title('PDF Document Query and Analysis App')

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    st.subheader(f"Processing: {uploaded_file.name}")
    
    with st.spinner('Converting PDF to images...'):
        temp_file_path = save_uploadedfile(uploaded_file)
        doc = fitz.open(temp_file_path)
        output_dir = tempfile.mkdtemp()

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            output = os.path.join(output_dir, f"page{page_num + 1}.png")
            pix.save(output)
        doc.close()

    st.success('PDF converted to images successfully!')

    with st.spinner('Extracting information from images...'):
        folder_path = Path(output_dir)
        markdown_content = ""
        image_paths = []
        for file_path in sorted(folder_path.iterdir()):
            page_num = int(file_path.stem.replace('page', ''))
            markdown_content += f"\n## Page {page_num}\n"
            markdown_content += get_generated_data(str(file_path))
            image_paths.append((page_num, str(file_path)))

    with st.spinner('Storing data in vector database...'):
        md_file_path = os.path.join(tempfile.mkdtemp(), "extracted_content.md")
        with open(md_file_path, "w") as f:
            f.write(markdown_content)

        loader = UnstructuredMarkdownLoader(md_file_path)
        data = loader.load()

        vector_db = Milvus.from_documents(
            data,
            embeddings,
            connection_args=MILVUS_CONNECTION_ARGS,
        )

    st.session_state['vector_db'] = vector_db
    st.session_state['image_paths'] = image_paths
    st.session_state['document_name'] = uploaded_file.name
    st.success("PDF processed and stored in vector database!")

    with st.expander("View Extracted Content"):
        st.markdown(markdown_content)

# Query interface
st.subheader("Query the Document")
query = st.text_input("Enter your query about the document:")
if st.button("Search"):
    if 'vector_db' in st.session_state:
        with st.spinner('Searching...'):
            docs = st.session_state['vector_db'].similarity_search(query, k=5)  # Retrieve top 5 relevant chunks
            content = "\n".join([f"Page {doc.metadata['page_number']}: {doc.page_content}" for doc in docs])

            system_content = "You are an assisting agent. Please provide the response based on the input."
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

            st.subheader("Relevant Pages:")
            for doc in docs:
                st.write(f"Page {doc.metadata['page_number']}")
    else:
        st.warning("Please upload and process a PDF first.")

# Display images below the search box
if 'image_paths' in st.session_state:
    st.subheader(f"Pages from: {st.session_state['document_name']}")
    for page_num, img_path in sorted(st.session_state['image_paths']):
        with st.expander(f"Page {page_num}"):
            st.image(img_path, use_column_width=True)
else:
    st.info("Upload a PDF to see extracted images.")
