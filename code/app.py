from flask import Flask, request, jsonify, Response
from videodetection import DetectionModel
from livedetection import process_frame
import os
import cv2

app = Flask(__name__)
model_instance = DetectionModel()

if not os.path.exists("./temp/"):
    os.makedirs("./temp/")

@app.route("/", methods=["GET"])
def index():
    return """<!DOCTYPE html>
  <html>
  <body>
    <a href='http://localhost:8501'>Go to prediction app</a>
  </body>
  </html>"""

@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded!"})
    
    video_fs = request.files["video"]
    filename = video_fs.filename
    temp_filepath = os.path.join("./input_videos/", filename)
    
    video_fs.save(temp_filepath)
    
    output_path = model_instance.detect_shoplifting(temp_filepath, filename)

    video_info = model_instance._utils.get_video_info(output_path)

    return jsonify({"output_path": output_path, "video_info": video_info})

@app.route('/live_stream')
def live_stream():
    stream_url = request.args.get('url', '')
    if not stream_url:
        return "No stream URL provided", 400

    def generate_frames():
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            raise ValueError(f"Unable to open stream {stream_url}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
if __name__ == "__main__":
    app.run(debug=True)