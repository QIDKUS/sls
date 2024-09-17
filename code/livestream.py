from flask import Flask, Response
import cv2

app = Flask(__name__)

def generate_frames(video_path):
    while True:  # Loop indefinitely to keep the stream running
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                cap.release()  # Release the capture and break the loop
                break

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # Change 'shoplift3.mp4' to your actual video path
    return Response(generate_frames('input_videos\shoplift2.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)
