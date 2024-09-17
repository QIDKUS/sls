# Shoplifting Detection System

This project implements a shoplifting detection system using computer vision and machine learning techniques. It provides both video processing and live stream detection capabilities.

## Features

1. **Video Upload and Processing**:
   Users can upload video files (MP4 or AVI format) through the web interface. The system processes the video using a sophisticated sequence model to detect potential shoplifting incidents. Processed videos are saved with visual indicators of detected shoplifting probabilities, and high-confidence segments are extracted as separate clips.

2. **Live Stream Detection**:
   The system supports real-time shoplifting detection on live video streams. Users can input a stream URL through the web interface, and the system uses a YOLO-based model to detect and highlight potential shoplifting activities in real-time. Visual indicators are added to the stream, showing bounding boxes and confidence scores.

3. **User-Friendly Web Interface**:
   Built with Streamlit, the interface offers an intuitive user experience with easy-to-use controls for uploading videos, starting live streams, and viewing results. It provides progress indicators during processing and options to download processed videos and view extracted clips of high-confidence detections.

4. **Flexible Backend API**:
   A RESTful API built with Flask allows for easy integration with other systems or custom frontends. It supports both video file processing and live stream analysis, making it versatile for various application scenarios.

5. **Advanced Machine Learning Models**:
   The system employs state-of-the-art machine learning models for accurate detection. It uses a combination of feature extraction (MobileNetV3Large) and sequence modeling (GRU-based RNN) for video analysis, and YOLO (You Only Look Once) object detection for real-time stream processing.


Watch the demo video of the output clip
<video src='assets/democlip.mp4'/>

## API Endpoints

1. **Root Endpoint**
   - URL: `/`
   - Method: GET
   - Description: Returns a simple HTML page with a link to the main prediction application.
   - Response: HTML content

2. **Video Prediction Endpoint**
   - URL: `/predict`
   - Method: POST
   - Description: Processes an uploaded video file for shoplifting detection.
   - Request:
     - Body: Form-data with a "video" field containing the video file (MP4 or AVI format)
   - Response: JSON object containing:
     - `output_path`: Path to the processed video file
     - `video_info`: Information about the processed video (e.g., frame count, duration)
   - Error Response:
     - `{"error": "No video uploaded!"}` if no video file is provided

3. **Live Stream Processing Endpoint**
   - URL: `/live_stream`
   - Method: GET
   - Description: Processes a live video stream for real-time shoplifting detection.
   - Query Parameters:
     - `url`: The URL of the live stream to process
   - Response: 
     - A multipart response containing a series of JPEG images representing the processed video frames.
     - Each frame includes visual indicators for detected shoplifting activities.
   - Error Response:
     - Status code 400 if no stream URL is provided

## Project Structure

- `app.py`: Flask backend server
- `ui.py`: Streamlit frontend interface
- `videodetection.py`: Video processing and shoplifting detection logic
- `livedetection.py`: Live stream processing and detection
- `config/parameters.py`: Configuration parameters (not provided in the code snippets)
- `models/`: Directory containing pre-trained models
  - `ckpt.weights.h5`: Weights for the video detection
  - `live_weights.pt`: YOLO model weights for live detection

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/CodisteTechTeam/Shoplift_Detection
   cd  code
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the necessary model weights in the `models/` directory.

## Usage

1. Start the Flask backend server:
   ```
   python app.py
   ```

2. In a separate terminal, run the Streamlit frontend:
   ```
   streamlit run ui.py
   ```

3. Open your web browser and navigate to `http://localhost:8501` to access the user interface.

## Model Information

The system uses two main models:

1. Sequence Model (for video processing):
   - Architecture: GRU-based RNN
   - Input: Sequence of frame features extracted using MobileNetV3Large
   - Output: Shoplifting probability

2. YOLO Model (for live detection):
   - Architecture: YOLOv8 (assumed based on the usage of Ultralytics YOLO)
   - Input: Video frames
   - Output: Bounding boxes and class probabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).

