import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import load_model

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="😊",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 25px;
        font-weight: bold;
        border: 1px solid #ddd;
        margin-top: 20px;
    }

    .info-box {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">😊 Emotion Detection from Facial Images</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI / Deep Learning Based Facial Emotion Recognition</div>',
    unsafe_allow_html=True
)

# =========================================================
# EMOTION LABELS
# =========================================================

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = Path("static") / "model" / "emotion_model.h5"

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_emotion_model():

    if not MODEL_PATH.exists():
        st.error(
            f"❌ Model file not found:\n\n{MODEL_PATH}"
        )
        st.stop()

    try:
        return load_model(str(MODEL_PATH))
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


model = load_emotion_model()

# =========================================================
# FACE DETECTOR
# =========================================================

# =========================================================
# FACE DETECTOR
# =========================================================

CASCADE_FILE = Path("haarcascade_frontalface_default.xml")

if not CASCADE_FILE.exists():
    st.error(
        "❌ haarcascade_frontalface_default.xml file not found."
    )
    st.stop()

try:
    face_detector = cv2.CascadeClassifier(
        str(CASCADE_FILE)
    )

    if face_detector.empty():
        st.error("❌ Face detector could not be loaded.")
        st.stop()

except Exception as e:
    st.error(f"❌ Face detector error: {e}")
    st.stop()
# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_face(face):

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (48, 48))

    normalized = resized.astype("float32") / 255.0

    reshaped = np.reshape(
        normalized,
        (1, 48, 48, 1)
    )

    return reshaped


# =========================================================
# PREDICT EMOTION
# =========================================================

def predict_emotion(face):

    processed_face = preprocess_face(face)

    prediction = model.predict(
        processed_face,
        verbose=0
    )[0]

    emotion_index = int(np.argmax(prediction))

    emotion = emotion_labels[emotion_index]

    confidence = float(prediction[emotion_index]) * 100

    return emotion, confidence, prediction


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Settings")

option = st.sidebar.radio(
    "Select Detection Mode",
    [
        "Image Upload",
        "Webcam"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Supported Emotions**

    😠 Angry  
    🤢 Disgust  
    😨 Fear  
    😊 Happy  
    😐 Neutral  
    😢 Sad  
    😲 Surprise
    """
)

# =========================================================
# IMAGE UPLOAD
# =========================================================

if option == "Image Upload":

    st.header("📷 Upload Facial Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        image_array = np.array(image)

        image_bgr = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        output_image = image_bgr.copy()

        if len(faces) == 0:

            st.warning(
                "⚠️ No face detected. Please upload a clear facial image."
            )

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        else:

            st.success(
                f"✅ {len(faces)} face(s) detected!"
            )

            results = []

            for i, (x, y, w, h) in enumerate(faces):

                face = image_bgr[
                    y:y+h,
                    x:x+w
                ]

                try:

                    emotion, confidence, prediction = (
                        predict_emotion(face)
                    )

                    results.append(
                        (emotion, confidence)
                    )

                    cv2.rectangle(
                        output_image,
                        (x, y),
                        (x+w, y+h),
                        (0, 255, 0),
                        2
                    )

                    label = (
                        f"{emotion} "
                        f"{confidence:.1f}%"
                    )

                    cv2.putText(
                        output_image,
                        label,
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                except Exception as e:

                    st.error(
                        f"Prediction error: {e}"
                    )

            output_rgb = cv2.cvtColor(
                output_image,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                output_rgb,
                caption="Emotion Detection Result",
                use_container_width=True
            )

            # =============================================
            # RESULTS
            # =============================================

            if results:

                st.subheader("🎯 Prediction Results")

                for i, (emotion, confidence) in enumerate(results):

                    st.markdown(
                        f"""
                        <div class="result-box">
                            Face {i + 1}: {emotion}
                            <br>
                            <small>
                                Confidence: {confidence:.2f}%
                            </small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(
                        min(confidence / 100, 1.0)
                    )


# =========================================================
# WEBCAM
# =========================================================

elif option == "Webcam":

    st.header("🎥 Webcam Emotion Detection")

    st.info(
        "Allow camera permission in your browser."
    )

    camera_image = st.camera_input(
        "Take a picture"
    )

    if camera_image is not None:

        image = Image.open(
            camera_image
        ).convert("RGB")

        image_array = np.array(image)

        image_bgr = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        output_image = image_bgr.copy()

        if len(faces) == 0:

            st.warning(
                "⚠️ No face detected."
            )

            st.image(
                image,
                caption="Camera Image",
                use_container_width=True
            )

        else:

            results = []

            for x, y, w, h in faces:

                face = image_bgr[
                    y:y+h,
                    x:x+w
                ]

                try:

                    emotion, confidence, prediction = (
                        predict_emotion(face)
                    )

                    results.append(
                        (emotion, confidence)
                    )

                    cv2.rectangle(
                        output_image,
                        (x, y),
                        (x+w, y+h),
                        (0, 255, 0),
                        2
                    )

                    label = (
                        f"{emotion} "
                        f"{confidence:.1f}%"
                    )

                    cv2.putText(
                        output_image,
                        label,
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                except Exception as e:

                    st.error(
                        f"Prediction error: {e}"
                    )

            output_rgb = cv2.cvtColor(
                output_image,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                output_rgb,
                caption="Webcam Emotion Result",
                use_container_width=True
            )

            if results:

                st.subheader("🎯 Emotion Result")

                for emotion, confidence in results:

                    st.markdown(
                        f"""
                        <div class="result-box">
                            {emotion}
                            <br>
                            <small>
                                Confidence:
                                {confidence:.2f}%
                            </small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(
                        min(confidence / 100, 1.0)
                    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        <p>🤖 Emotion Detection using Deep Learning</p>
        <p>Developed using Python, TensorFlow, OpenCV and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)