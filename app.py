
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from functools import wraps
from datetime import datetime
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from src.helper import download_hugging_face_embeddings
from src.prompt import *
from store_index import process_files, check_file_content_extractable  # Add check_file_content_extractable
from pinecone_setup import initialize_pinecone
from pinecone import Pinecone

import secrets
import hashlib
import os
import shutil
import json
import time
import tempfile

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
app.secret_key = secrets.token_hex(16)

app.config['UPLOAD_FOLDER'] = 'Data'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'json', 'txt', 'docx', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD')




print("Initializing Pinecone...")
index_name = "event"
pc = initialize_pinecone(index_name=index_name)

embeddings_model = download_hugging_face_embeddings()

try:
    print(f"Attempting to connect to existing index '{index_name}'...")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings_model
    )
    print("Successfully connected to Pinecone index!")
except ValueError as e:
    print(f"Error connecting to index: {e}")

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})




print("Initializing LLM...")
llm = ChatGroq(
    temperature=0.4,
    max_tokens=500,
    model_name="llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("Chatbot setup complete!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session or not session['admin_logged_in']:
            flash('Please log in to access the admin panel', 'danger')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    # print("===User query===:",input)
    response = rag_chain.invoke({"input": msg})
    # print("Respponse printed   ===============", response)
    return str(response["answer"])

# ---------------- REST OF YOUR EXISTING CODE UNCHANGED ----------------


# --- Clear Chat ---

# Add this with your other route imports at the top
from datetime import datetime

# Add this with your other session management code
SESSION_STORE = {}  # In-memory session store (for production, use Redis or database)

# Add this route to your Flask app
@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'status': 'error', 'message': 'Session ID required'}), 400
        
        # Store the cleared timestamp (you could also store chat history here if needed)
        SESSION_STORE[session_id] = {
            'last_cleared': datetime.now().isoformat(),
            'history': []  # This would store chat history if you implement it
        }
        
        return jsonify({
            'status': 'success',
            'message': 'Chat history cleared',
            'cleared_at': SESSION_STORE[session_id]['last_cleared']
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# --- Admin Auth Routes ---
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
     

        if username == ADMIN_USERNAME and password== ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            flash('Login successful', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('admin_login'))


from datetime import datetime

# Register a custom Jinja2 filter
@app.template_filter('timestamp_to_datetime')
def timestamp_to_datetime_filter(timestamp):
    try:
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return 'Invalid Timestamp'


# --- Admin Dashboard ---
@app.route('/admin', methods=['GET'])
@admin_required
def admin_dashboard():
    files = []
    metadata_dict = get_all_file_metadata()
    
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        # Skip metadata.json and directories
        if filename == 'metadata.json' or os.path.isdir(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
            continue
            
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path) / 1024
            file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            # Get processed status from metadata
            processed = False
            for item in metadata_dict:
                if item.get('filename') == filename:
                    processed = item.get('processed', False)
                    break
                    
            files.append({
                'name': filename,
                'size': f"{file_size:.2f} KB",
                'date': file_date.strftime("%Y-%m-%d %H:%M:%S"),
                'processed': processed
            })
    
    return render_template('admin_dashboard.html', files=files, stats={})


# --- File Upload ---
@app.route('/admin/upload', methods=['POST'])
@admin_required
def upload_file():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('admin_dashboard'))

    file = request.files['file']

    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('admin_dashboard'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Check if file already exists
        if os.path.exists(file_path):
            flash(f'File {filename} already exists. Please rename your file or delete the existing one.', 'warning')
            return redirect(url_for('admin_dashboard'))
        
        file.save(file_path)

        file_metadata = {
            'filename': filename,
            'processed': False,
            'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        save_file_metadata(file_metadata)
        
        flash(f'File {filename} uploaded successfully! Click on Process button to add it to the knowledge base.', 'success')
        return redirect(url_for('admin_dashboard'))

    flash('File type not allowed', 'danger')
    return redirect(url_for('admin_dashboard'))





@app.route('/admin/process/<filename>', methods=['POST'])
@admin_required
def process_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        flash(f'File {filename} not found', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    # Check if file is already processed
    metadata = get_file_metadata(filename)
    if metadata and metadata.get('processed', False):
        flash(f'File {filename} is already processed!', 'info')
        return redirect(url_for('admin_dashboard'))
    
    try:
        # Create a temporary directory for processing
        new_data_folder = os.path.join('Data', 'temp')
        os.makedirs(new_data_folder, exist_ok=True)
        temp_file_path = os.path.join(new_data_folder, filename)
        
        # Copy the file to the temporary directory
        shutil.copy2(file_path, temp_file_path)

        # First, check if the file has extractable content
        print(f"Checking if content can be extracted from {filename}...")
        
        # Call the enhanced process_files function
        result = process_files(data_dir=new_data_folder, index_name='event')

        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(new_data_folder) and not os.listdir(new_data_folder):
            os.rmdir(new_data_folder)

        # Check the result
        if result and hasattr(result, 'processed_chunks') and result.processed_chunks > 0:
            # Update processed status in metadata
            update_file_processed_status(filename, True)
            flash(f'File {filename} processed successfully and added to knowledge base! Extracted {result.processed_chunks} text chunks.', 'success')
        elif hasattr(result, 'success') and not result.success:
            # This is our new ProcessingResult object
            if "no meaningful content" in result.error_message.lower() or "scanned images" in result.error_message.lower():
                flash(f'No content could be extracted from {filename}. The file may contain only scanned images or unsupported content. OCR processing was attempted but insufficient text was found.', 'warning')
            else:
                flash(f'Error processing file {filename}: {result.error_message}', 'danger')
        else:
            # Fallback for backward compatibility
            flash(f'No content could be extracted from {filename}. The file may contain only scanned images or unsupported content. OCR processing was attempted but insufficient text was found.', 'warning')
            
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'danger')
        print(f"Exception in process_file: {str(e)}")
        import traceback
        traceback.print_exc()

    return redirect(url_for('admin_dashboard'))



# --- Helper functions for file metadata ---
def get_all_file_metadata():
    """Get all file metadata entries"""
    metadata_file = os.path.join(app.config['UPLOAD_FOLDER'], 'metadata.json')
    
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
            return all_metadata
        else:
            return []
    except Exception as e:
        app.logger.error(f"Error getting all file metadata: {str(e)}")
        return []

def get_file_metadata(filename):
    """Get metadata for a specific file"""
    all_metadata = get_all_file_metadata()
    
    for item in all_metadata:
        if item.get('filename') == filename:
            return item
            
    return None

def save_file_metadata(metadata):
    metadata_file = os.path.join(app.config['UPLOAD_FOLDER'], 'metadata.json')
    
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = []
        
        for i, item in enumerate(all_metadata):
            if item['filename'] == metadata['filename']:
                all_metadata[i] = metadata
                break
        else:
            all_metadata.append(metadata)
        
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
            
    except Exception as e:
        app.logger.error(f"Error saving file metadata: {str(e)}")

def update_file_processed_status(filename, processed=True):
    metadata_file = os.path.join(app.config['UPLOAD_FOLDER'], 'metadata.json')
    
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = []
        
        updated = False
        for i, item in enumerate(all_metadata):
            if item['filename'] == filename:
                all_metadata[i]['processed'] = processed
                all_metadata[i]['process_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                updated = True
                break
        
        # If the file wasn't found in metadata, add it
        if not updated:
            all_metadata.append({
                'filename': filename,
                'processed': processed,
                'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'process_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
            
    except Exception as e:
        app.logger.error(f"Error updating file processed status: {str(e)}")

def get_unprocessed_files():
    metadata_file = os.path.join(app.config['UPLOAD_FOLDER'], 'metadata.json')
    unprocessed = []
    
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
            
            unprocessed = [item['filename'] for item in all_metadata if not item.get('processed', False)]
            
    except Exception as e:
        app.logger.error(f"Error getting unprocessed files: {str(e)}")
        
    return unprocessed


@app.route('/admin/delete/<filename>')
@admin_required
def delete_file(filename):
    print(f"Attempting to delete file: {filename}")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # --- Delete embeddings from Pinecone ---
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index("event")
        
        # Use consistent fileName filter
        filter_obj = {"fileName": filename}
        
        try:
            # Delete vectors with fileName filter
            index.delete(filter=filter_obj)
            print(f"Deleted vectors from Pinecone for file: {filename}")
            flash(f'Successfully removed embeddings for {filename} from Pinecone', 'success')
        except Exception as e:
            print(f"Error deleting from Pinecone: {str(e)}")
            flash(f"Error deleting embeddings: {str(e)}", "danger")

    except Exception as e:
        app.logger.error(f"Error connecting to Pinecone: {str(e)}")
        flash(f"Error connecting to Pinecone: {str(e)}", "danger")

    # --- Delete file locally and update metadata ---
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            remove_file_metadata(filename)
            flash(f'File {filename} deleted successfully from both database and local storage', 'success')
        except Exception as e:
            app.logger.error(f"Error deleting local file {filename}: {str(e)}")
            flash(f"Failed to delete file {filename} locally", "danger")
    else:
        # File doesn't exist locally, but we may have deleted from database
        remove_file_metadata(filename)
        flash(f'File {filename} removed from database (local file was already missing)', 'info')

    return redirect(url_for('admin_dashboard'))

 # --- Remove file metadata ---............................................................................

def remove_file_metadata(filename):
    """Remove a file from metadata.json"""
    metadata_file = os.path.join(app.config['UPLOAD_FOLDER'], 'metadata.json')
    
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
            
            # Filter out the file to be removed
            all_metadata = [item for item in all_metadata if item.get('filename') != filename]
            
            with open(metadata_file, 'w') as f:
                json.dump(all_metadata, f, indent=2)
                
    except Exception as e:
        app.logger.error(f"Error removing file metadata: {str(e)}")

# === Main Entry Point ===
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)



