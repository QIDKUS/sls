import os
import cv2
import time
from keras.applications import MobileNetV3Large
from keras.applications.mobilenet_v3 import preprocess_input
from keras.layers import GRU, Dense, Dropout, Input
from keras import Model
import tensorflow as tf


class Utils:
    def __init__(self) -> None:
        self.frame_size = (224, 224)
        self.img_size = 224

    def get_video_frames_from_bytes(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames

    def cut_video_chunks(self, video_frames, chunk_size):
        frames = []
        chunk = []
        for frame in video_frames:
            chunk.append(frame)
            if len(chunk) == chunk_size:
                frames.append(chunk)
                chunk = []
        return frames

    def stitch_video(self, processed_frames, output_path):
        height, width, _ = processed_frames[0][0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, 24.0, (width, height))
        for frames in processed_frames:
            for frame in frames:
                writer.write(frame)
        writer.release()

    def get_video_info(self, video_path):
        cap = cv2.VideoCapture(video_path)
        video_info = {
            "path": video_path,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        return video_info

    def write_prediction(self, frame, prediction):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img=frame,
            text=f"SHOPLIFTING: {prediction} %",
            org=(10, 20),
            fontFace=font,
            fontScale=0.75,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_4,
        )
        return frame

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = cv2.resize(frame, dsize=self.frame_size)  # Resize
        frame = (frame - frame.min()) / (frame.max() - frame.min())
        return frame


class DetectionModel:
    def __init__(self) -> None:
        self._utils = Utils()
        self.MAX_SEQ_LENGTH = 30
        self.NUM_FEATURES = 960
        self._model_weights_filepath = "models/ckpt.weights.h5"
        self.feature_extractor = self.build_feature_extractor()
        self.model = self.get_sequence_model()
        self.model.load_weights(self._model_weights_filepath)

    def get_sequence_model(self):
        frame_features_input = Input((self.MAX_SEQ_LENGTH, self.NUM_FEATURES))

        x = GRU(32, return_sequences=True)(frame_features_input)
        x = GRU(16)(x)
        x = Dropout(0.4)(x)
        x = Dense(8, activation="relu")(x)
        output = Dense(1, activation="sigmoid")(x)

        rnn_model = Model(frame_features_input, output)

        rnn_model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return rnn_model

    def build_feature_extractor(self):
        img_size = self._utils.img_size
        feature_extractor = MobileNetV3Large(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(img_size, img_size, 3),
        )

        inputs = Input((img_size, img_size, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        return Model(inputs, outputs, name="feature_extractor")

    def get_features(self, video_frames):
        frame_features = []
        for frame in video_frames:
            features = self.feature_extractor.predict(frame[None, :], verbose=0)
            features = tf.squeeze(features, axis=0)
            frame_features.append(features)
            all_features = tf.stack(frame_features, axis=0)
        return all_features

    def get_prediction(self, frames):
        preprocessed_frames = [self._utils.preprocess_frame(frame) for frame in frames]
        frame_features = self.get_features(preprocessed_frames)
        prediction = self.model.predict(frame_features[None, :], verbose=0)
        prediction = tf.concat(prediction, axis=1)
        return round(prediction.numpy()[0][0] * 100, 2)


    def detect_shoplifting(self, video_path, filename):

        video_frames = self._utils.get_video_frames_from_bytes(video_path)
        video_info = self._utils.get_video_info(video_path)
        frame_count = video_info['frame_count']
        fps = 24

        file_name = filename.split('.')[0]

        if not os.path.exists(f"./{file_name}/"):
            os.makedirs(f"./{file_name}/")

        output_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"{file_name}"
        )
        clips_folder = os.path.join(output_folder, "clips_folder")
        output_video_folder = os.path.join(output_folder, "output_video")

        if not os.path.exists(clips_folder):
            os.makedirs(clips_folder)

        if not os.path.exists(output_video_folder):
            os.makedirs(output_video_folder)

        processed_chunks = []

        chunk_size = 30
        chunks = self._utils.cut_video_chunks(video_frames, chunk_size)
        segment_count = 0 
        prediction_frames = []
        clip_list = []

        last_index = None

        for chunk_idx, chunk in enumerate(chunks):
            processed_chunk = []
            prediction = self.get_prediction(chunk)
            for frame in chunk:
                frame_with_prediction = self._utils.write_prediction(frame, prediction)
                prediction_frames.append(frame_with_prediction)
                processed_chunk.append(frame_with_prediction)
            processed_chunks.append(processed_chunk)
            if prediction > 90:
                print(chunk_idx)
                if(last_index == None or (chunk_idx-last_index) >= 3):
                    start_frame_idx = max(chunk_idx * chunk_size - 3 * fps, 0)
                    end_frame_idx = min((chunk_idx + 1) * chunk_size + 3 * fps, frame_count)
                    clip_data = [start_frame_idx,end_frame_idx]
                    clip_list.append(clip_data)
                    last_index = chunk_idx
                else:
                    continue
            else:
                continue
        
        for row in range(0,len(clip_list)):
            start_frame_idx = clip_list[row][0]
            end_frame_idx = clip_list[row][1]
            segment_frames = prediction_frames[start_frame_idx:end_frame_idx]
            timestamp_seconds = start_frame_idx / fps
            minutes, seconds = divmod(int(timestamp_seconds), 60)
            timestamp = f"{minutes:02d}_{seconds:02d}"

            segment_count += 1
            high_confidence_output_path = os.path.join(
                clips_folder,
                f"shopliftclip_segment_{segment_count}_at_{timestamp}.mp4"
            )

            height, width, _ = segment_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(high_confidence_output_path, fourcc, fps, (width, height))

            for frame in segment_frames:
                writer.write(frame)  

            writer.release()


        output_path = os.path.join(output_video_folder, f"output_{file_name}.mp4")
        self._utils.stitch_video(processed_chunks, output_path)

        return output_path




    