import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_milvus.vectorstores import Milvus
import fitz  # PyMuPDF for handling PDFs
import base64
import os
from pathlib import Path

# Set the API key using st.secrets for secure access
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
MODEL = "gpt-4o"
client = OpenAI()
embeddings = OpenAIEmbeddings()

# Define the function to handle image data and send it to OpenAI API
def get_generated_data(image_path):
    """Encode image and send request to OpenAI for content extraction."""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    SYSTEM_PROMPT = "You are a helpful assistant that responds in Markdown. Help me with Given Image Extraction with Given Details with Different categories!"
    USER_PROMPT = "Retrieve all the information provided in the image, including figures, titles, and graphs."

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"}
            ]}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

# Streamlit interface
st.title('PDF Document Query and Analysis App')

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    # Convert PDF pages to images for extraction
    st.subheader("PDF Processing and Image Extraction")
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    output_dir = "./data/output/"
    os.makedirs(output_dir, exist_ok=True)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        output = os.path.join(output_dir, f"page{page_num + 1}.png")
        pix.save(output)

    doc.close()

    # Process each image for data extraction
    folder_path = Path(output_dir)
    markdown_content = ""
    for file_path in folder_path.iterdir():
        markdown_content += "\n" + get_generated_data(str(file_path))

    # Display extracted markdown content
    st.text_area("Extracted Content:", markdown_content, height=300)

    # Perform text query on extracted data
    query = st.text_input("Enter your query about the extracted data:")
    if st.button("Search"):
        with st.spinner('Searching...'):
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an assisting agent. Please provide the response based on the input."},
                    {"role": "user", "content": f"Respond to the query '{query}' using the information from the following content: {markdown_content}"}
                ]
            )
            st.write(response.choices[0].message.content)
