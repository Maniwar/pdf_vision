import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI, OpenAIError
import base64
from io import BytesIO
from PIL import Image  # Import PIL for image handling
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import Milvus as LangchainMilvus
from langchain_openai import OpenAIEmbeddings

# Set the OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]

# Initialize the OpenAI client and embeddings model
client = OpenAI(api_key=openai_api_key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize Milvus cloud connection
connections.connect(
    alias="default",
    uri=st.secrets["general"]["MILVUS_PUBLIC_ENDPOINT"],
    secure=True,
    token=st.secrets["general"]["MILVUS_API_KEY"]
)

# Create a Milvus collection
collection_name = "pdf_embeddings"
if not utility.has_collection(collection_name):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
    ]
    schema = CollectionSchema(fields, description="PDF embeddings collection")
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(name=collection_name)

# Constants
SYSTEM_PROMPT = "You are a helpful assistant that responds in Markdown. Help me with Given Image Extraction with Given Details with Different categories!"
USER_PROMPT = """
Retrieve all the information provided in the image, including figures, titles, and graphs.
"""

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_images_from_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img_data = BytesIO(pix.tobytes(output="png"))
        img = Image.open(img_data)
        images.append(img)
    return images

def generate_embeddings(image_base64):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
                {"role": "user", "content": f"data:image/png;base64,{image_base64}"}
            ]
        )
        return response.choices[0].message.content  # Ensure correct field access
    except OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        return None

def main():
    st.title("Medical Document Assistant")

    st.sidebar.header("Options")
    
    # Instructions for doctors
    st.write("### Instructions")
    st.write("""
    1. Upload your medical documents in PDF format.
    2. The system will convert each page into an image and generate embeddings.
    3. You can query the uploaded documents to retrieve specific information.
    """)

    # Document upload and embedding
    st.sidebar.subheader("Upload Documents")
    uploaded_files = st.sidebar.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            images = extract_images_from_pdf(uploaded_file)

            for i, image in enumerate(images):
                image_base64 = encode_image(image)
                embedding = generate_embeddings(image_base64)
                if embedding:
                    data = [
                        [i],  # id
                        [embedding],  # Ensure embedding is a list if not already
                        [f"Page {i+1} of {uploaded_file.name}"]  # Text field
                    ]
                    collection.insert(data)
                    st.session_state.embeddings.append(f"{uploaded_file.name}_page_{i+1}")
                    st.sidebar.write(f"Processed and stored embeddings for {uploaded_file.name}_page_{i+1}")

    # Display current embeddings
    st.sidebar.write("### Current Embeddings")
    st.sidebar.write(st.session_state.embeddings)

    # Chat with data
    st.write("### Query the Documents")
    query = st.text_input("Enter your query here:")
    if query:
        langchain_milvus = LangchainMilvus(collection, embeddings)
        docs = langchain_milvus.similarity_search(query, k=5)
        response_content = "Top results:\n"
        for doc in docs:
            response_content += f"Document ID: {doc.id}\n"
            response_content += f"Score: {doc.score}\n"
            response_content += f"Content: {doc.metadata['text']}\n\n"
        st.write(response_content)

if __name__ == "__main__":
    main()
