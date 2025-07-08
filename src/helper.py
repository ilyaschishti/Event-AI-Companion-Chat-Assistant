

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import pytesseract
from PIL import Image
import pdf2image
import cv2
import numpy as np
import os
import glob
import json
import tempfile
import shutil

# OCR Configuration - Update this path based on your Tesseract installation
# Windows example: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Linux/Mac: usually just 'tesseract' (default)
pytesseract.pytesseract.tesseract_cmd = 'tesseract'  # Adjust this path as needed

def preprocess_image_for_ocr(image):
    """
    Preprocess image to improve OCR accuracy
    """
    # Convert PIL Image to OpenCV format
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Apply threshold to get better contrast
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def extract_text_from_image_ocr(image_path):
    """
    Extract text from a single image using OCR
    """
    try:
        # Load image
        image = Image.open(image_path)
        
        # Preprocess image
        processed_image = preprocess_image_for_ocr(image)
        
        # Convert back to PIL Image for pytesseract
        processed_pil = Image.fromarray(processed_image)
        
        # Extract text using OCR
        text = pytesseract.image_to_string(processed_pil, lang='eng')
        
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from image {image_path}: {str(e)}")
        return ""

def extract_text_from_pdf_ocr(pdf_path):
    """
    Extract text from PDF using OCR (for scanned PDFs)
    """
    try:
        # Convert PDF pages to images
        pages = pdf2image.convert_from_path(pdf_path, dpi=300)
        
        extracted_text = ""
        
        for page_num, page in enumerate(pages):
            print(f"Processing page {page_num + 1} of {len(pages)} with OCR...")
            
            # Preprocess the page image
            processed_image = preprocess_image_for_ocr(page)
            processed_pil = Image.fromarray(processed_image)
            
            # Extract text using OCR
            page_text = pytesseract.image_to_string(processed_pil, lang='eng')
            
            if page_text.strip():
                extracted_text += f"\n--- Page {page_num + 1} ---\n"
                extracted_text += page_text.strip() + "\n"
        
        return extracted_text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF with OCR {pdf_path}: {str(e)}")
        return ""

def load_image_files_ocr(data_dir):
    """
    Load and extract text from image files using OCR
    """
    documents = []
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    for ext in image_extensions:
        image_files = glob.glob(os.path.join(data_dir, ext))
        image_files.extend(glob.glob(os.path.join(data_dir, ext.upper())))
        
        for image_path in image_files:
            try:
                print(f"Processing image file: {image_path}")
                text = extract_text_from_image_ocr(image_path)
                
                if text and len(text.strip()) > 10:  # Only if meaningful text is extracted
                    metadata = {
                        "source": image_path,
                        
                        "fileName": os.path.basename(image_path)
                    }
                    
                    documents.append(Document(page_content=text, metadata=metadata))
                    print(f"Extracted {len(text)} characters from {os.path.basename(image_path)}")
                else:
                    print(f"No meaningful text found in {os.path.basename(image_path)}")
                    
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
    
    return documents

def load_pdf_file_enhanced(data_dir):
    """
    Enhanced PDF loader that tries regular extraction first, then OCR if needed
    """
    documents = []
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    for pdf_path in pdf_files:
        try:
            print(f"Processing PDF: {pdf_path}")
            
            # First, try regular PDF extraction
            loader = PyPDFLoader(pdf_path)
            regular_docs = loader.load()
            
            # Check if meaningful text was extracted
            total_text = " ".join([doc.page_content for doc in regular_docs])
            
            if len(total_text.strip()) > 50:  # If substantial text was extracted
                print(f"Regular text extraction successful for {os.path.basename(pdf_path)}")
                documents.extend(regular_docs)
            else:
                print(f"Regular extraction failed for {os.path.basename(pdf_path)}, trying OCR...")
                
                # Try OCR extraction
                ocr_text = extract_text_from_pdf_ocr(pdf_path)
                
                if ocr_text and len(ocr_text.strip()) > 50:
                    metadata = {
                        "source": pdf_path,
                       
                        "fileName": os.path.basename(pdf_path)
                    }
                    
                    documents.append(Document(page_content=ocr_text, metadata=metadata))
                    print(f"OCR extraction successful for {os.path.basename(pdf_path)}")
                else:
                    print(f"No meaningful text could be extracted from {os.path.basename(pdf_path)}")
                    
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
    
    return documents

def load_docx_file_enhanced(data_dir):
    """
    Enhanced DOCX loader that tries regular extraction first, then OCR if needed
    """
    documents = []
    docx_files = glob.glob(os.path.join(data_dir, "*.docx"))
    
    for docx_path in docx_files:
        try:
            print(f"Processing DOCX: {docx_path}")
            
            # First, try regular DOCX extraction
            loader = UnstructuredWordDocumentLoader(docx_path)
            regular_docs = loader.load()
            
            # Check if meaningful text was extracted
            total_text = " ".join([doc.page_content for doc in regular_docs])
            
            if len(total_text.strip()) > 50:  # If substantial text was extracted
                print(f"Regular text extraction successful for {os.path.basename(docx_path)}")
                documents.extend(regular_docs)
            else:
                print(f"Regular extraction failed for {os.path.basename(docx_path)}")
                # Note: For DOCX with embedded images, you'd need more complex processing
                # This is a placeholder for future enhancement
                
        except Exception as e:
            print(f"Error processing DOCX {docx_path}: {str(e)}")
    
    return documents

# Original functions (keeping for backward compatibility)
def load_pdf_file(data_dir):
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

def load_json_file(data_dir):
    import json
    from langchain_core.documents import Document
    import glob
    import os

    documents = []
    json_files = glob.glob(os.path.join(data_dir, "*.json"))

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                chunks = data.get("chunks", [])
                
                for chunk in chunks:
                    content = chunk.get("content", "").strip()
                    if content:
                        metadata = {
                            "source": file_path,
                          
                            "fileName": os.path.basename(file_path)
                        }
                        
                        documents.append(Document(page_content=content, metadata=metadata))

            print(f"Loaded JSON file: {file_path} with {len(chunks)} chunks")
        except Exception as e:
            print(f"Error loading JSON file {file_path}: {str(e)}")

    return documents

def load_txt_file(data_dir):
    loader = DirectoryLoader(
        data_dir,
        glob="*.txt",
        loader_cls=lambda path: TextLoader(path, encoding='utf-8', autodetect_encoding=True)
    )
    return loader.load()

def load_docx_file(data_dir):
    loader = DirectoryLoader(
        data_dir,
        glob="*.docx",
        loader_cls=UnstructuredWordDocumentLoader
    )
    return loader.load()

def load_all_files_enhanced(data_dir):
    """
    Enhanced version that includes OCR capabilities
    """
    all_documents = []

    loaders = [
        ("PDF (Enhanced with OCR)", load_pdf_file_enhanced),
        ("Images (OCR)", load_image_files_ocr),
        ("JSON", load_json_file),
        ("TXT", load_txt_file),
        ("DOCX (Enhanced)", load_docx_file_enhanced)
    ]

    for filetype, loader_func in loaders:
        try:
            docs = loader_func(data_dir)
            all_documents.extend(docs)
            print(f"Loaded {len(docs)} {filetype} documents")
        except Exception as e:
            print(f"Error loading {filetype} documents: {str(e)}")

    return all_documents

# Keep original function for backward compatibility
def load_all_files(data_dir):
    """
    Original function - now calls the enhanced version
    """
    return load_all_files_enhanced(data_dir)

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(extracted_data)

def download_hugging_face_embeddings():
    embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings_model