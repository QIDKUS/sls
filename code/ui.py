import streamlit as st
import requests
import os
import cv2
import numpy as np

BACKEND_URL = "http://127.0.0.1:5000"
TIMEOUT = 120

def get_video_info(uploaded_video):
    if uploaded_video is not None:
        filename = uploaded_video.name
        video_name = filename.split('.')[0]
        clips_folder = os.path.join(os.getcwd(), f"{video_name}", "clips_folder")
        return filename, video_name, clips_folder
    return None, None, None

# Set page config
st.set_page_config(page_title="Shoplifting Detection", page_icon="🛡️", layout="wide")

# Custom CSS with improved visibility
st.markdown("""
    <style>
    .main {
        background-color: #2C3E50;
        color: #ECF0F1;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #2980B9;
    }
    .stRadio>div {
        padding: 10px;
        background-color: #34495E;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stRadio>div>label {
        color: #ECF0F1 !important;
    }
    .stSelectbox>div>div>div {
        background-color: #34495E;
        color: #ECF0F1;
    }
    h1, h2, h3 {
        color: #3498DB;
    }
    .stTextInput>div>div>input {
        background-color: #34495E;
        color: #ECF0F1;
    }
    .custom-header {
        background-color: black;
        color: white;
        padding: 20px 0;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


# Initialize session state variables
if "detection_mode" not in st.session_state:
    st.session_state.detection_mode = None
if "uploaded_video" not in st.session_state:
    st.session_state.uploaded_video = None
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "live_stream_active" not in st.session_state:
    st.session_state.live_stream_active = False

st.markdown("""
    <div class="custom-header">
        Shoplifting Detection
    </div>
    """, unsafe_allow_html=True)

# Step 1: Choose detection mode
st.subheader("Step 1: Choose Detection Mode")
detection_mode = st.radio("", ("Upload Video", "Live Stream"), key="detection_mode")

if detection_mode != st.session_state.detection_mode:
    st.session_state.detection_mode = detection_mode
    st.session_state.uploaded_video = None
    st.session_state.prediction_done = False
    st.session_state.live_stream_active = False

if detection_mode == "Upload Video":
    st.subheader("Step 2: Upload Video")
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi"], key="video_uploader")

    if uploaded_video is not None:
        st.session_state.uploaded_video = uploaded_video
        filename, video_name, clips_folder = get_video_info(st.session_state.uploaded_video)
        st.success(f"Successfully uploaded: {filename}")
        
        st.subheader("Step 3: Process Video")
        if st.button("Start Detection", key="predict_button") and not st.session_state.prediction_done:
            with st.spinner("Processing video... This may take a few minutes."):
                with open(filename, "wb") as f:
                    f.write(st.session_state.uploaded_video.getbuffer())
                
                files = {"video": (filename, st.session_state.uploaded_video.read())}
                
                response = requests.post(f"{BACKEND_URL}/predict", files=files)
                result = response.json()
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    output_path = result["output_path"]
                    st.session_state.output_path = output_path
                    st.session_state.video_info = result.get("video_info")
                    st.session_state.prediction_done = True
                    st.success("Detection complete! 🎉")
                    
                    st.subheader("Step 4: Download Processed Video")
                    with open(output_path, "rb") as video_bytes:
                        st.download_button(
                            label="Download Processed Video",
                            data=video_bytes,
                            file_name=f"processed_{filename}",
                            mime="video/mp4",
                        )
                    st.session_state.clips_folder = clips_folder

        if st.session_state.prediction_done:
            st.subheader("Step 5: View Detected Clips")

            if os.path.exists(st.session_state.clips_folder):
                clips = [f for f in os.listdir(st.session_state.clips_folder) if f.endswith(('.mp4', '.avi'))]
                if clips:
                    selected_clip = st.selectbox(
                        "Select a clip to view:",
                        clips,
                        key="clip_selector"
                    )

                    clip_path = os.path.join(st.session_state.clips_folder, selected_clip)
                    st.video(clip_path)
                else:
                    st.info("No suspicious clips detected in this video.")
            else:
                st.info("No clips folder found. This might be due to no shoplifting incidents detected.")

elif detection_mode == "Live Stream":
    st.subheader("Live Stream Shoplifting Detection")
    stream_link = st.text_input("Enter the stream link:", key="stream_link_input")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Live Stream", key="start_stream_button") and stream_link:
            st.session_state.live_stream_active = True

    if st.session_state.get("live_stream_active", False):
        stframe = st.empty()
        with col2:
            stop_button = st.button("Stop Live Stream", key="stop_stream_button")

        try:
            with requests.get(f"{BACKEND_URL}/live_stream", params={"url": stream_link}, stream=True, timeout=TIMEOUT) as response:
                if response.status_code == 200:
                    bytes_data = bytes()
                    for chunk in response.iter_content(chunk_size=1024):
                        bytes_data += chunk
                        a = bytes_data.find(b'\xff\xd8')
                        b = bytes_data.find(b'\xff\xd9')

                        if a != -1 and b != -1:
                            jpg = bytes_data[a:b+2]
                            bytes_data = bytes_data[b+2:]
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            stframe.image(frame, channels="BGR", use_column_width=True)

                        if stop_button:
                            st.session_state.live_stream_active = False
                            st.success("Live stream stopped.")
                            break
                else:
                    st.error(f"Error: Received status code {response.status_code} from the server")
        except requests.exceptions.Timeout:
            st.error(f"Timeout error: The request took longer than {TIMEOUT} seconds")
            st.session_state.live_stream_active = False
        except requests.exceptions.RequestException as e:
            st.error(f"Error in live stream: {e}")
            st.session_state.live_stream_active = False
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.session_state.live_stream_active = False