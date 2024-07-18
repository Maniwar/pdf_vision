import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI, OpenAIError
import base64
from io import BytesIO
from PIL import Image  # Import PIL for image handling
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from langchain_community.document_loaders import UnstructuredMarkdownLoader
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

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_pdf_and_extract_embeddings(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img_data = BytesIO(pix.tobytes(output="png"))
        img = Image.open(img_data)
        image_base64 = encode_image(img)
        embedding = generate_embeddings(image_base64)
        if embedding:
            # Prepare data for Milvus
            data = {
                "id": page_num,
                "embedding": [embedding],
                "text": f"Page {page_num + 1} of {file.name}"
            }
            # Insert data into Milvus
            collection.insert([data])
            st.session_state.embeddings.append(data)

def generate_embeddings(image_base64):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract data from image."},
                {"role": "user", "content": f"data:image/png;base64,{image_base64}"}
            ]
        )
        return response.choices[0].message['content']
    except OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        return None

def main():
    st.title("Document Processing with AI")
    st.sidebar.header("Options")
    
    # Initialize session state
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = []

    uploaded_file = st.file_uploader("Choose PDF files", type=["pdf"])
    if uploaded_file:
        process_pdf_and_extract_embeddings(uploaded_file)

    # Display embeddings and other data
    st.write("### Current Embeddings")
    for embedding in st.session_state.embeddings:
        st.write(embedding)

if __name__ == "__main__":
    main()
