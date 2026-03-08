# ASR Audio Transcription Web App

A powerful web application for Automatic Speech Recognition (ASR) built using **Flask** and **OpenAI's Whisper** models. This platform allows users to upload audio files and receive highly accurate, timestamped text transcriptions.

## Core Features

- **Multilingual Support**: Optimized specifically for multiple languages including English, Tamil, Hindi, Telugu, Malayalam, and Kannada.
- **Hardware Acceleration**: Automatically detects and utilizes NVIDIA CUDA (GPUs) for drastically faster processing using mixed-precision (fp16) logic when available. Fallbacks gracefully to CPU with lighter models.
- **Dynamic Resource Handling**: Selects the optimal Whisper model (`small` or `medium`) depending on the availability of GPU capabilities, language selections, and desired performance.
- **Real-Time Progress Tracking**: Users can track the status of the transcription dynamically via the `/progress` endpoint.
- **Automated Audio Normalization**: Uses `ffmpeg` under the hood to normalize all ingested audio into 16kHz mono WAV formats to ensure highest compatibility and accuracy during inference.
- **Language-Specific Optimizations**: Uses specially crafted custom prompts and decoding parameters for localized Indian languages (like Tamil) to increase transcription accuracy and reduce hallucination.

## Technologies Used

- **Backend Context**: Python 3.10+, Flask, Flask-CORS
- **Machine Learning**: PyTorch, OpenAI Whisper
- **Audio Processing**: FFmpeg
- **Frontend**: HTML5, CSS3, JavaScript (handled via `index.html`)

## Gallery

### 1. English Song Transcription
![English Song Transcription](Images/English%20song.png)

### 2. Tamil Song Transcription (Example 1)
![Tamil Song Transcription 1](Images/Tamil%20song%201.png)

### 3. Tamil Song Transcription (Example 2)
![Tamil Song Transcription 2](Images/Tamil%20song%202.png)

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tarzan6250/asr_audio.git
   cd asr_audio
   ```

2. **Environment & Dependencies**
   It's recommended to set up a virtual environment. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have `ffmpeg` installed on your system and available in your system's PATH.*

3. **Run the Application**
   ```bash
   python app.py
   ```
   The application will be accessible at `http://localhost:5000/`.

## License
All rights reserved.
