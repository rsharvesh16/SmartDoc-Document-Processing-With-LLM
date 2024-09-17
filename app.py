import streamlit as st
import fitz
import pytesseract
from pdf2image import convert_from_path
import nltk
import re
import os
from dotenv import load_dotenv
import boto3
import tempfile
import json
import ssl

# Load Environment Variables
load_dotenv()

# Pytesseract Configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Ensure NLTK data is downloaded
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set up AWS Bedrock client
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")

if aws_access_key_id and aws_secret_access_key:
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
else:
    st.error("AWS credentials not found. Please set the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")

# Function to convert PDF to text
def convert_pdf_to_txt(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error in converting PDF to text: {e}")
        return ""

# Function to extract text from images using OCR
def extract_text_from_images(images):
    try:
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error in OCR: {e}")
        return ""

# Function to preprocess text
def preprocess_text(text):
    try:
        def clean_text(text):
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        cleaned_text = clean_text(text)
        sentences = nltk.sent_tokenize(cleaned_text)
        return sentences
    except Exception as e:
        st.error(f"Error in preprocessing text: {e}")
        return []

# Function to chunk text
def chunk_text(text, max_chunk_size=4000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    for word in words:
        if current_size + len(word) + 1 > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# Function to call AWS Bedrock Mistral Large LLM
def generate_response(prompt):
    try:
        chunks = chunk_text(prompt)
        responses = []
        for chunk in chunks:
            body = json.dumps({
                "prompt": chunk,
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
            })
            modelId = "mistral.mistral-7b-instruct-v0:2"
            response = bedrock.invoke_model(body=body, modelId=modelId)
            response_body = json.loads(response.get('body').read())
            responses.append(response_body.get('outputs')[0].get('text'))
        return ' '.join(responses)
    except Exception as e:
        st.error(f"Error in generating response from AWS Bedrock: {e}")
        return "Unable to generate response due to an error."

# Main Streamlit interface
st.title("SmartDoc: Intelligent Document Processing with AWS Bedrock Integration")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    # Convert PDF to text
    text = convert_pdf_to_txt(temp_file_path)

    # OCR on images
    full_text = text
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            pages = convert_from_path(temp_file_path, 500)
            image_text = extract_text_from_images(pages)
            full_text += "\n" + image_text
    except Exception as e:
        st.warning(f"Unable to extract images from PDF. OCR will not be performed. Error: {e}")

    # Preprocess text
    preprocessed_text = preprocess_text(full_text)

    # Information Extraction
    if aws_access_key_id and aws_secret_access_key:
        entities = generate_response(f"Extract entities from the following text: {full_text[:4000]}\n\nNames, dates, locations, organizations.")
        relationships = generate_response(f"Extract relationships between entities from the following text: {full_text[:4000]}")
        summary = generate_response(f"Summarize the following text: {full_text[:4000]}")
        categories = ["Personal", "Work", "Legal", "Medical"]
        classification = generate_response(f"Classify the following text into one of these categories: {categories}\n\nText: {full_text[:4000]}")
    else:
        entities = relationships = summary = classification = "AWS credentials not set. Unable to perform this operation."

    st.header("SmartDoc - ChatBot")
    user_input = st.text_input("Enter a Question to ask..")
    if user_input and aws_access_key_id and aws_secret_access_key:
        prompt = f'''You are a helpful, respectful and honest assistant. Always answer as 
        helpfully as possible, while being safe. Your answers should not include
        any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
        Please ensure that your responses are socially unbiased and positive in nature.
        If a question does not make any sense, or is not factually coherent, explain 
        why instead of answering something not correct. If you don't know the answer 
        to a question, please don't share false information.
        Your goal is to provide {user_input} answers from the following text: {full_text[:4000]} accurately.'''

        ans = generate_response(prompt)
        st.text(ans)

    st.header("To know Information about your Document:")

    # Create dropdowns for displaying the information
    with st.expander("Extracted Text and OCR Extracted Text from Images (Combined)"):
        st.subheader("Extracted Text and OCR Extracted Text from Images (Combined)")
        st.text(full_text[:1000] + "..." if len(full_text) > 1000 else full_text)

    with st.expander("Preprocessed Text"):
        st.subheader("Preprocessed Text")
        st.write(preprocessed_text[:10])  # Display first 10 sentences

    with st.expander("Information Extraction"):
        st.subheader("Information Extraction")
        st.text("Entities:")
        st.text(entities)
        st.text("Relationships:")
        st.text(relationships)
        st.text("Summary:")
        st.text(summary)

    with st.expander("Document Classification"):
        st.subheader("Document Classification")
        st.text(f"Classification: {classification}")
    
    # Internal Translation
    if aws_access_key_id and aws_secret_access_key:
        target_language = st.selectbox("Select target language for translation", ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Tamil"])
        if target_language:
            translation = generate_response(f"Translate the following text to {target_language}: {full_text[:4000]}")
            with st.expander(f"Translation to {target_language}"):
                st.subheader(f"Translation to {target_language}")
                st.text(translation)

    # Clean up temporary files
    os.unlink(temp_file_path)