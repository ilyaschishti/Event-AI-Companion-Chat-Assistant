

from src.helper import load_all_files_enhanced, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os
import time
import re

def clean_text(text):
    """Clean text by removing unnecessary whitespace"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

class ProcessingResult:
    def __init__(self):
        self.processed_chunks = 0
        self.success = False
        self.error_message = ""
        self.files_processed = []
        self.files_failed = []

def process_files(data_dir='Data/', index_name='event'):
    result = ProcessingResult()
    
    try:
        # Check if index exists
        active_indexes = pc.list_indexes()
        print(f"Active indexes: {[index.name for index in active_indexes]}")

        if index_name not in [index.name for index in active_indexes]:
            print(f"Creating index {index_name}...")
            pc.create_index(
                name=index_name,
                dimension=384,  # Must match your embeddings dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to initialize
            print("Waiting for index to be ready...")
            time.sleep(60)

        # Load and process documents using enhanced loader
        print(f"Loading documents from {data_dir}...")
        extracted_data = load_all_files_enhanced(data_dir)
        
        # Clean text content
        for doc in extracted_data:
            doc.page_content = clean_text(doc.page_content)
        
        if not extracted_data:
            result.error_message = "No documents found to process or no meaningful content could be extracted."
            print(result.error_message)
            return result
            
        print(f"Loaded {len(extracted_data)} total documents")
        # Track which files were processed - UPDATE this section
        for doc in extracted_data:
            # Ensure fileName exists in metadata, fallback to extracting from source
            if 'fileName' not in doc.metadata and 'source' in doc.metadata:
                doc.metadata['fileName'] = os.path.basename(doc.metadata['source'])

            filename = doc.metadata.get('fileName', 'unknown')
            if filename not in result.files_processed:
                result.files_processed.append(filename)

                
        text_chunks = text_split(extracted_data)

        # Check if there are any chunks after splitting
        if not text_chunks or len(text_chunks) == 0:
            result.error_message = "No text chunks could be extracted from the documents. The files may contain only scanned images or unsupported content."
            print(result.error_message)
            return result

        
        for chunk in text_chunks:
         if 'source' in chunk.metadata:
            source_path = chunk.metadata['source']
            filename = os.path.basename(source_path)
            chunk.metadata = {'fileName': filename}  # Keep only filename   


        embeddings_model = download_hugging_face_embeddings()

        # Store embeddings in Pinecone
        print("Creating vector store...")
        docsearch = PineconeVectorStore.from_documents(
            documents=text_chunks,
            embedding=embeddings_model,
            index_name=index_name
        )

        # Set success metrics
        result.processed_chunks = len(text_chunks)
        result.success = True
        
        print(f"Successfully processed {len(text_chunks)} chunks into Pinecone index '{index_name}'")
        print(f"Files processed: {result.files_processed}")
        
        # Add processed chunks count to the return object for backward compatibility
        docsearch.processed_chunks = len(text_chunks)
        return docsearch

    except Exception as e:
        result.error_message = f"Error during processing: {str(e)}"
        result.success = False
        print(result.error_message)
        import traceback
        traceback.print_exc()
        return result

def check_file_content_extractable(file_path):
    """
    Check if meaningful content can be extracted from a file
    """
    try:
        temp_dir = os.path.dirname(file_path)
        extracted_data = load_all_files_enhanced(temp_dir)
        
        # Check if any content was extracted
        total_text = ""
        for doc in extracted_data:
            if doc.metadata.get('source') == file_path:
                total_text += doc.page_content
        
        return len(total_text.strip()) > 50  # Return True if substantial content found
        
    except Exception as e:
        print(f"Error checking file content: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    vector_store = process_files()
    if hasattr(vector_store, 'processed_chunks') and vector_store.processed_chunks > 0:
        print("Pinecone setup completed successfully!")
    else:
        print("Processing completed but no content was indexed.")