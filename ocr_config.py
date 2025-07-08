# """
# OCR Configuration File
# Configure Tesseract path and OCR settings here
# """

# import os
# import platform

# # Tesseract Configuration
# def get_tesseract_path():
#     """
#     Get the appropriate Tesseract path based on the operating system
#     """
#     system = platform.system().lower()
    
#     if system == "windows":
#         # Common Windows installation paths
#         possible_paths = [
#             r"C:\Program Files\Tesseract-OCR\tesseract.exe",
#             r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
#             r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', ''))
#         ]
        
#         for path in possible_paths:
#             if os.path.exists(path):
#                 return path
        
#         # If not found, return default and let user configure
#         return r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
#     elif system == "darwin":  # macOS
#         # Common macOS installation paths
#         possible_paths = [
#             "/usr/local/bin/tesseract",
#             "/opt/homebrew/bin/tesseract",
#             "/usr/bin/tesseract"
#         ]
        
#         for path in possible_paths:
#             if os.path.exists(path):
#                 return path
        
#         return "tesseract"  # Default to system PATH
    
#     else:  # Linux and other Unix-like systems
#         # Common Linux installation paths
#         possible_paths = [
#             "/usr/bin/tesseract",
#             "/usr/local/bin/tesseract",
#             "/bin/tesseract"
#         ]
        
#         for path in possible_paths:
#             if os.path.exists(path):
#                 return path
        
#         return "tesseract"  # Default to system PATH

# # OCR Settings
# OCR_CONFIG = {
#     'tesseract_path': get_tesseract_path(),
#     'language': 'eng',  # Default language
#     'dpi': 300,  # DPI for PDF to image conversion
#     'min_text_length': 10,  # Minimum text length to consider extraction successful
#     'image_preprocessing': True,  # Enable image preprocessing for better OCR
# }

# # Supported file extensions
# SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
# SUPPORTED_DOCUMENT_EXTENSIONS = ['.pdf', '.docx', '.txt', '.json']
# ALL_SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_DOCUMENT_EXTENSIONS