import streamlit as st
import fitz  # For PyMuPDF
import pytesseract  # For OCR (Pytesseract)
from pdf2image import convert_from_path  # For Pdf to Image Conversions
import nltk  # For Preprocessing Texts
import re  # For Preprocessing Texts
import os
from dotenv import load_dotenv  # For Loading Environment Variables
import google.generativeai as genai  # For Loading Gemini Pro LLM
import tempfile

# Load Environment Variables
load_dotenv()

# Pytesseract Configuration
pytesseract.pytesseract.tesseract_cmd = r'Your Tesseract Path'

# Ensure NLTK data is downloaded
nltk.download('punkt')

# Set Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

# Function to convert PDF to text
def convert_pdf_to_txt(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Function to extract text from images using OCR
def extract_text_from_images(images):
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

# Function to preprocess text
def preprocess_text(text):
    def clean_text(text):
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def segment_sentences(text):
        return nltk.sent_tokenize(text)
    
    def tokenize_sentences(sentences):
        return [nltk.word_tokenize(sentence) for sentence in sentences]

    cleaned_text = clean_text(text)
    sentences = segment_sentences(cleaned_text)
    tokenized_sentences = tokenize_sentences(sentences)
    return tokenized_sentences

# Function to call Gemini LLM
def generate_response(prompt):
    response = model.generate_content(prompt)
    return response.text

# Function to extract entities
def extract_entities(text):
    prompt = f"Extract entities from the following text: {text}\n\nNames, dates, locations, organizations."
    entities = generate_response(prompt)
    return entities

# Function to extract relationships
def extract_relationships(text):
    prompt = f"Extract relationships between entities from the following text: {text}"
    relationships = generate_response(prompt)
    return relationships

# Function to summarize text
def summarize_text(text):
    prompt = f"Summarize the following text: {text}"
    summary = generate_response(prompt)
    return summary

# Function to classify document
def classify_document(text, categories):
    prompt = f"Classify the following text into one of these categories: {categories}\n\nText: {text}"
    classification = generate_response(prompt)
    return classification

# Function to translate text
def translate_text(text, target_language):
    prompt = f"Translate the following text to {target_language}: {text}"
    translation = generate_response(prompt)
    return translation

# Streamlit interface
st.title("SmartDoc: Intelligent Document Processing with LLM Integration")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert PDF to text
    text = convert_pdf_to_txt("temp.pdf")

    # OCR on images
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            pages = convert_from_path("temp.pdf", 500, poppler_path=r'Your Poppler Path')
            image_text = extract_text_from_images(pages)
            full_text = text + "\n" + image_text
    except Exception as e:
        st.error(f"Error in extracting images from PDF: {e}")
        full_text = text

    # Preprocess text
    preprocessed_text = preprocess_text(full_text)

    # Information Extraction
    entities = extract_entities(full_text)
    relationships = extract_relationships(full_text)
    summary = summarize_text(full_text)

    # Document Classification
    categories = ["Personal", "Work", "Legal", "Medical"]
    classification = classify_document(full_text, categories)

    st.header("SmartDoc - ChatBot")
    user_input = st.text_input("Enter a Question to ask..")
    if user_input:
        prompt = f'''You are a helpful, respectful and honest assistant. Always answer as 
        helpfully as possible, while being safe. Your answers should not include
        any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
        Please ensure that your responses are socially unbiased and positive in nature.
        If a question does not make any sense, or is not factually coherent, explain 
        why instead of answering something not correct. If you don't know the answer 
        to a question, please don't share false information.
        Your goal is to provide {user_input} answers from the {full_text} accurately.'''

        ans = generate_response(prompt)
        st.text(ans)

    
    st.header("To know Information about your Document:")

    # Create dropdowns for displaying the information
    with st.expander("Extracted Text and OCR Extracted Text from Images (Combined)"):
        st.subheader("Extracted Text and OCR Extracted Text from Images (Combined)")
        st.text(full_text)

    with st.expander("Preprocessed Text"):
        st.subheader("Preprocessed Text")
        st.write(preprocessed_text)

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
    target_language = st.selectbox("Select target language for translation", ["English", "Spanish", "French", "German", "Chinese", "Japanese"])
    if target_language:
        translation = translate_text(full_text, target_language)

    if target_language:
        with st.expander(f"Translation to {target_language}"):
            st.subheader(f"Translation to {target_language}")
            st.text(translation)

    # Clean up temporary files
    os.remove("temp.pdf")
