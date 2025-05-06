from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import whisper
import os
import torch
import time
import uuid
import numpy as np
import json
import gc
import threading
import tempfile
import subprocess
import random
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Enable CORS with specific settings
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}})

# Global progress tracking
progress_data = {
    "status": "idle",
    "progress": 0,
    "message": "",
    "session_id": ""
}

# Force CUDA configuration if available
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU
    torch.cuda.set_device(0)  # Explicitly set PyTorch to use first GPU
    device = "cuda"
    print(f"[+] CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"[+] CUDA version: {torch.version.cuda}")
    # Clear GPU memory before loading model
    torch.cuda.empty_cache()
    gc.collect()
    # Set PyTorch to use TensorFloat-32 precision if available (for Ampere+ GPUs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable cudnn benchmark for faster convolutions
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"
    print("[!] CUDA is not available. Using CPU instead.")

# Load the appropriate model based on available resources
# For Tamil, we'll use a smaller model for faster processing
if device == "cuda":
    # On GPU, we can use medium model with fp16 for Tamil
    try:
        model = whisper.load_model("medium", device=device)
        # Try to enable fp16 safely
        if torch.cuda.is_available():
            # Check if the GPU supports fp16
            if torch.cuda.get_device_capability()[0] >= 7:  # Volta or newer architecture
                print(f"[+] GPU supports fp16, enabling mixed precision")
                # Enable automatic mixed precision
                torch.backends.cuda.matmul.allow_tf32 = True
            else:
                print(f"[+] GPU may not fully support fp16, using standard precision")
    except Exception as e:
        print(f"[!] Error loading model with fp16: {str(e)}")
        model = whisper.load_model("medium", device=device)
    
    print(f"[+] Loaded Whisper 'medium' model on {device}")
else:
    # On CPU, use small model for faster processing
    model = whisper.load_model("small", device=device)
    print(f"[+] Loaded Whisper 'small' model on {device} for faster processing")

print(f"[+] Model is using optimized precision for multilingual support")

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    # Serve the index.html file with proper HTML content type
    try:
        response = send_from_directory("", "index.html")
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        return response
    except Exception as e:
        print(f"[!] Error serving index.html: {str(e)}")
        # Return a simple HTML response if file not found
        html = "<html><body><h1>Error loading application</h1><p>Please check server logs.</p></body></html>"
        response = make_response(html)
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        return response

# Custom error handler for all exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    print(f"[!] Unhandled exception: {str(e)}")
    
    # Return JSON instead of HTML for HTTP errors
    response = make_response(json.dumps({
        "error": str(e),
        "status": "error"
    }), 500)
    response.headers['Content-Type'] = 'application/json'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# Add response headers to ensure proper JSON responses
@app.after_request
def add_header(response):
    # Only set Content-Type for API routes, not for static files
    # Check if the response doesn't already have a content type set
    if (request.path.startswith('/progress') or request.path.startswith('/transcribe')) and \
       not response.headers.get('Content-Type'):
        response.headers['Content-Type'] = 'application/json'
    
    # Don't override HTML content type
    if request.path == '/' and not response.headers.get('Content-Type'):
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route("/progress", methods=['GET', 'OPTIONS'])
def get_progress():
    global progress_data
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
    
    # Create a proper JSON response with explicit headers
    response = make_response(json.dumps(progress_data))
    response.headers['Content-Type'] = 'application/json'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Generate a unique session ID for this transcription
    session_id = str(uuid.uuid4())
    
    # Reset progress tracking
    global progress_data
    progress_data = {
        "status": "processing",
        "progress": 0,
        "message": "Starting transcription...",
        "session_id": session_id
    }

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Update progress
    progress_data["progress"] = 10
    progress_data["message"] = "File uploaded, preparing audio..."

    # Get language preference from request, default to English if not specified
    language = request.form.get('language', 'en')
    
    # Set model size based on language for better performance
    model_size = "small"
    if language == "ta" and device == "cpu":  # Tamil on CPU
        model_size = "small"  # Use smaller model for Tamil on CPU for speed
        print(f"[+] Using 'small' model for Tamil on CPU for faster processing")
    elif language == "ta" and device == "cuda":  # Tamil on GPU
        model_size = "medium"  # Medium is a good balance for Tamil on GPU
        print(f"[+] Using 'medium' model for Tamil on GPU")
    else:  # Other languages
        model_size = "medium" if device == "cuda" else "small"
        print(f"[+] Using '{model_size}' model for {language}")
    
    print(f"[+] Transcribing: {filepath}")
    print(f"[+] Using language: {language}")
    
    try:
        # Load the audio file
        audio = whisper.load_audio(filepath)
        
        # Update progress
        progress_data["progress"] = 20
        progress_data["message"] = "Audio loaded, normalizing..."
        
        # Use absolute paths to avoid path-related errors
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create a unique filename based on session ID
        temp_wav_path = os.path.join(temp_dir, f"temp_{session_id}.wav")
        
        # Convert and normalize audio using ffmpeg
        try:
            # Use absolute paths and explicit encoding for Tamil and other languages
            subprocess.run(['ffmpeg', '-y', '-i', filepath, '-ar', '16000', '-ac', '1', 
                           '-sample_fmt', 's16', '-vn', '-c:a', 'pcm_s16le', temp_wav_path], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"[+] Audio normalized and converted to 16kHz mono WAV")
            
            # Verify the file exists and has content
            if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
                # Use the normalized audio file instead
                filepath = temp_wav_path
                
                # Force the specified language
                print(f"[+] Forcing language: {language}")
                
                # Reload audio after normalization
                audio = whisper.load_audio(filepath)
            else:
                print(f"[!] Converted file is empty or doesn't exist, using original audio")
        except Exception as e:
            print(f"[!] Error during audio conversion: {str(e)}")
            # If conversion fails, use the original audio
            print(f"[+] Using original audio file without conversion")
            
            # Update progress
            progress_data["progress"] = 30
            progress_data["message"] = "Audio normalized, preparing model..."
        except Exception as e:
            print(f"[!] Error during audio conversion: {str(e)}")
            # If conversion fails, use the original audio
            print(f"[+] Using original audio file without conversion")
            
            # Update progress
            progress_data["progress"] = 30
            progress_data["message"] = "Using original audio, preparing model..."
        
        # Clear GPU memory before transcription
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        # Create a model instance optimized for the specific language
        progress_data["progress"] = 40
        progress_data["message"] = "Loading transcription model..."
        
        transcription_model = whisper.load_model(model_size, device=device)
        print(f"[+] Created {model_size} transcription model for {language}")
        
        # For Tamil specifically, optimize decoding parameters
        if language == "ta":
            print(f"[+] Applying Tamil-specific optimizations")
        
        progress_data["progress"] = 50
        progress_data["message"] = "Model loaded, starting transcription..."
        
        # Set a random seed for reproducibility within this session
        random_seed = random.randint(1, 10000)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        
        print(f"[+] Using random seed: {random_seed} for unique transcription")
        
        # Transcribe with the specified language
        progress_data["progress"] = 60
        progress_data["message"] = "Transcribing audio..."
        
        # Prepare language-specific prompts to improve transcription
        initial_prompt = None
        if language == "ta":
            # Tamil-specific prompt to help with transcription - user provided
            initial_prompt = "இந்த பாடல் வரிகள் தமிழ் மொழியில் உள்ளன. தயவு செய்து துல்லியமாக டைப்செய்யவும்."
            print(f"[+] Using custom Tamil prompt for better accuracy")
        elif language == "hi":
            initial_prompt = "यह एक गाने के बोल हैं। कृपया हिंदी में सही से लिखें।"
        elif language == "te":
            initial_prompt = "ఇది ఒక పాట సాహిత్యం. దయచేసి తెలుగులో సరిగ్గా రాయండి."
        
        # Optimize parameters based on language
        try:
            # Use a try/except block to catch any errors during transcription
            if language == "ta":  # Tamil-specific optimizations for speed
                # Tamil language optimizations - faster processing
                result = transcription_model.transcribe(
                    audio,  # Use the loaded audio directly
                    word_timestamps=True,
                    language=language,
                    beam_size=1,  # Smallest beam size for fastest processing
                    fp16=(device == "cuda"),  # Use fp16 on GPU for Tamil
                    temperature=0.0,  # Use greedy decoding for better accuracy
                    without_timestamps=False,  # Ensure we get timestamps
                    condition_on_previous_text=False,  # Don't condition on previous text
                    initial_prompt=initial_prompt,  # Tamil-specific prompt
                    best_of=1,  # Only consider the top candidate to save time
                    patience=1.0  # Lower patience for faster decoding
                )
                print(f"[+] Using optimized settings for Tamil to reduce transcription time")
            elif language in ["hi", "te", "ml", "kn"]:  # Other Indian languages
                # Indian language optimizations
                result = transcription_model.transcribe(
                    audio,  # Use the loaded audio directly
                    word_timestamps=True,
                    language=language,
                    beam_size=3,  # Smaller beam size for faster processing
                    fp16=(device == "cuda"),  # Use fp16 on GPU for Indian languages
                    temperature=0.0,  # Use greedy decoding for better accuracy
                    without_timestamps=False,  # Ensure we get timestamps
                    condition_on_previous_text=False,  # Don't condition on previous text
                    initial_prompt=initial_prompt  # Language-specific prompt
                )
            else:
                # Standard parameters for other languages
                result = transcription_model.transcribe(
                    audio,  # Use the loaded audio directly
                    word_timestamps=True,
                    language=language,
                    beam_size=5,
                    fp16=(device == "cuda"),  # Use fp16 on GPU for speed
                    temperature=0.2,  # Small temperature for some variety
                    without_timestamps=False,  # Ensure we get timestamps
                    initial_prompt=None  # No initial prompt needed
                )
        except Exception as e:
            # If there's an error with the optimized parameters, try with safer defaults
            print(f"[!] Error during transcription with optimized parameters: {str(e)}")
            print(f"[+] Retrying with safer parameters...")
            
            # Fall back to safer parameters
            result = transcription_model.transcribe(
                audio,
                word_timestamps=True,
                language=language,
                beam_size=1,  # Simplest beam search
                fp16=False,  # Disable fp16
                temperature=0.0,  # Deterministic output
                without_timestamps=False
            )
        
        progress_data["progress"] = 80
        progress_data["message"] = "Transcription complete, processing results..."
        
        # Clear the model from memory
        del transcription_model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Extract segments with timestamps and language info
        progress_data["progress"] = 90
        progress_data["message"] = "Formatting results..."
        
        segments = []
        for segment in result["segments"]:
            segments.append({
                "start": format_time(segment["start"]),
                "end": format_time(segment["end"]),
                "text": segment["text"].strip()
            })
        
        response_data = {
            "segments": segments,
            "full_text": result["text"],
            "language": language
        }
        
        # Create a proper JSON response with explicit headers
        response = make_response(json.dumps(response_data))
        response.headers['Content-Type'] = 'application/json'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
        
    except Exception as e:
        print(f"[!] Error: {str(e)}")
        # Create a proper error response with explicit JSON formatting
        error_response = {"error": str(e)}
        
        # Create a proper JSON response with explicit headers
        response = make_response(json.dumps(error_response), 500)
        response.headers['Content-Type'] = 'application/json'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    finally:
        # Clean up all temporary files
        try:
            # Clean up the original uploaded file
            original_filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.exists(original_filepath):
                os.remove(original_filepath)
                print(f"[+] Removed original file: {original_filepath}")
            
            # Clean up the temporary WAV file if it exists
            if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                    print(f"[+] Removed temporary WAV file: {temp_wav_path}")
                except PermissionError:
                    print(f"[!] Could not remove temporary file due to permission error. Will be cleaned up later.")
                except Exception as e:
                    print(f"[!] Error removing temporary file: {str(e)}")
        except Exception as e:
            print(f"[!] Error during cleanup: {str(e)}")
            # Continue execution even if cleanup fails
    
    return jsonify(response)

def format_time(seconds):
    """Format seconds into MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

if __name__ == "__main__":
    app.run(debug=True)
