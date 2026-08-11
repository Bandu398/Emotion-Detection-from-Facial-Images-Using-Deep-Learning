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
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #172554 50%,
            #312e81 100%
        );
    }

    /* Remove top spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Main title */
    .title {
        text-align: center;
        font-size: 44px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 19px;
        color: #c7d2fe;
        margin-bottom: 8px;
    }

    /* Developer */
    .author {
        text-align: center;
        font-size: 18px;
        color: #a5b4fc;
        margin-bottom: 35px;
    }

    /* Main cards */
    .card {
        background: rgba(15, 23, 42, 0.88);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #6366f1;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.30);
        margin-top: 20px;
    }

    /* Result card */
    .result {
        background: rgba(30, 41, 59, 0.95);
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #818cf8;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.20);
        margin-top: 15px;
    }

    /* Emotion text */
    .emotion {
        font-size: 30px;
        font-weight: 700;
        color: #ffffff;
    }

    /* Confidence */
    .confidence {
        font-size: 18px;
        color: #cbd5e1;
        margin-top: 5px;
    }

    /* Section heading */
    h1, h2, h3 {
        color: #ffffff !important;
    }

    /* Normal text */
    p, label {
        color: #e2e8f0 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(
            90deg,
            #4f46e5,
            #7c3aed
        );
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 25px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #6366f1,
            #8b5cf6
        );
        color: white;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.75);
        border-radius: 15px;
        padding: 10px;
        border: 1px solid #475569;
    }

    /* Camera */
    [data-testid="stCameraInput"] {
        border-radius: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #c7d2fe;
        margin-top: 50px;
        padding: 25px;
        border-top: 1px solid #475569;
        font-size: 15px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111827,
            #1e1b4b
        );
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    """
    <div class="title">
        😊 Emotion Detection from Facial Images
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI / Deep Learning Based Facial Emotion Recognition
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="author">
        👨‍💻 Developed by <b>Anuj Bhoir</b>
    </div>
    """,
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

emotion_icons = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😊",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲"
}

# =========================================================
# MODEL PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "static"
    / "model"
    / "emotion_model.h5"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_emotion_model():

    if not MODEL_PATH.exists():

        st.error(
            f"""
            ❌ Model file not found.

            Expected location:

            {MODEL_PATH}
            """
        )

        st.stop()

    try:

        model = load_model(
            str(MODEL_PATH),
            compile=False
        )

        return model

    except Exception as e:

        st.error(
            f"""
            ❌ Error loading emotion model:

            {e}
            """
        )

        st.stop()


model = load_emotion_model()

# =========================================================
# HAAR CASCADE FACE DETECTOR
# =========================================================

CASCADE_FILE = (
    BASE_DIR
    / "haarcascade_frontalface_default.xml"
)

if not CASCADE_FILE.exists():

    st.error(
        """
        ❌ Haar Cascade file not found.

        Please make sure:

        haarcascade_frontalface_default.xml

        is in the same folder as app.py.
        """
    )

    st.stop()

# Check OpenCV
if not hasattr(cv2, "CascadeClassifier"):

    st.error(
        """
        ❌ OpenCV installation problem.

        CascadeClassifier is not available.

        Please check requirements.txt.
        """
    )

    st.stop()

try:

    face_detector = cv2.CascadeClassifier(
        str(CASCADE_FILE)
    )

except Exception as e:

    st.error(
        f"""
        ❌ Face detector error:

        {e}
        """
    )

    st.stop()

if face_detector.empty():

    st.error(
        """
        ❌ Haar Cascade could not be loaded.

        Please check the XML file.
        """
    )

    st.stop()

# =========================================================
# PREPROCESS FACE
# =========================================================

def preprocess_face(face):

    gray = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    resized = cv2.resize(
        gray,
        (48, 48)
    )

    normalized = (
        resized.astype("float32") / 255.0
    )

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

    emotion_index = int(
        np.argmax(prediction)
    )

    emotion = emotion_labels[
        emotion_index
    ]

    confidence = (
        float(prediction[emotion_index])
        * 100
    )

    return (
        emotion,
        confidence,
        prediction
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    <h2 style="color:white;">
    ⚙️ Settings
    </h2>
    """,
    unsafe_allow_html=True
)

mode = st.sidebar.radio(
    "Select Detection Mode",
    [
        "📷 Image Upload",
        "🎥 Webcam"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <h3 style="color:white;">
    🎭 Supported Emotions
    </h3>
    """,
    unsafe_allow_html=True
)

st.sidebar.write("😠 Angry")
st.sidebar.write("🤢 Disgust")
st.sidebar.write("😨 Fear")
st.sidebar.write("😊 Happy")
st.sidebar.write("😐 Neutral")
st.sidebar.write("😢 Sad")
st.sidebar.write("😲 Surprise")

st.sidebar.markdown("---")

st.sidebar.info(
    """
    This application uses:

    • Python  
    • TensorFlow  
    • OpenCV  
    • CNN  
    • Streamlit
    """
)

# =========================================================
# IMAGE UPLOAD MODE
# =========================================================

if mode == "📷 Image Upload":

    st.markdown(
        """
        <div class="card">
        """,
        unsafe_allow_html=True
    )

    st.header("📷 Upload Facial Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # IMAGE PROCESSING
    # -----------------------------------------------------

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_array = np.array(
            image
        )

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

        # -------------------------------------------------
        # NO FACE
        # -------------------------------------------------

        if len(faces) == 0:

            st.warning(
                """
                ⚠️ No face detected.

                Please upload a clear facial image.
                """
            )

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        # -------------------------------------------------
        # FACE FOUND
        # -------------------------------------------------

        else:

            st.success(
                f"✅ {len(faces)} face(s) detected!"
            )

            results = []

            for i, (
                x,
                y,
                w,
                h
            ) in enumerate(faces):

                face = image_bgr[
                    y:y+h,
                    x:x+w
                ]

                try:

                    emotion, confidence, prediction = (
                        predict_emotion(face)
                    )

                    results.append(
                        (
                            emotion,
                            confidence,
                            prediction
                        )
                    )

                    # Face rectangle
                    cv2.rectangle(
                        output_image,
                        (x, y),
                        (x+w, y+h),
                        (0, 255, 0),
                        3
                    )

                    # Label
                    label = (
                        f"{emotion} "
                        f"{confidence:.1f}%"
                    )

                    cv2.putText(
                        output_image,
                        label,
                        (
                            x,
                            max(y - 10, 25)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                except Exception as e:

                    st.error(
                        f"Prediction error: {e}"
                    )

            # -------------------------------------------------
            # DISPLAY RESULT IMAGE
            # -------------------------------------------------

            output_rgb = cv2.cvtColor(
                output_image,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                output_rgb,
                caption="🎯 Emotion Detection Result",
                use_container_width=True
            )

            # -------------------------------------------------
            # RESULTS
            # -------------------------------------------------

            if results:

                st.subheader(
                    "🎯 Prediction Results"
                )

                for i, (
                    emotion,
                    confidence,
                    prediction
                ) in enumerate(results):

                    icon = emotion_icons.get(
                        emotion,
                        "😊"
                    )

                    st.markdown(
                        f"""
                        <div class="result">

                            <div class="emotion">
                                {icon} Face {i + 1}: {emotion}
                            </div>

                            <div class="confidence">
                                Confidence:
                                {confidence:.2f}%
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(
                        min(
                            confidence / 100,
                            1.0
                        )
                    )

# =========================================================
# WEBCAM MODE
# =========================================================

else:

    st.markdown(
        """
        <div class="card">
        """,
        unsafe_allow_html=True
    )

    st.header(
        "🎥 Webcam Emotion Detection"
    )

    st.info(
        "Allow camera permission in your browser."
    )

    camera_image = st.camera_input(
        "Take a picture"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # CAMERA PROCESSING
    # -----------------------------------------------------

    if camera_image is not None:

        image = Image.open(
            camera_image
        ).convert("RGB")

        image_array = np.array(
            image
        )

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

        # -------------------------------------------------
        # NO FACE
        # -------------------------------------------------

        if len(faces) == 0:

            st.warning(
                "⚠️ No face detected."
            )

            st.image(
                image,
                caption="Camera Image",
                use_container_width=True
            )

        # -------------------------------------------------
        # FACE FOUND
        # -------------------------------------------------

        else:

            st.success(
                f"✅ {len(faces)} face(s) detected!"
            )

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
                        (
                            emotion,
                            confidence,
                            prediction
                        )
                    )

                    # Face box
                    cv2.rectangle(
                        output_image,
                        (x, y),
                        (x+w, y+h),
                        (0, 255, 0),
                        3
                    )

                    label = (
                        f"{emotion} "
                        f"{confidence:.1f}%"
                    )

                    cv2.putText(
                        output_image,
                        label,
                        (
                            x,
                            max(y - 10, 25)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                except Exception as e:

                    st.error(
                        f"Prediction error: {e}"
                    )

            # -------------------------------------------------
            # DISPLAY WEBCAM RESULT
            # -------------------------------------------------

            output_rgb = cv2.cvtColor(
                output_image,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                output_rgb,
                caption="🎯 Webcam Emotion Result",
                use_container_width=True
            )

            # -------------------------------------------------
            # EMOTION RESULT
            # -------------------------------------------------

            if results:

                st.subheader(
                    "🎯 Emotion Result"
                )

                for (
                    emotion,
                    confidence,
                    prediction
                ) in results:

                    icon = emotion_icons.get(
                        emotion,
                        "😊"
                    )

                    st.markdown(
                        f"""
                        <div class="result">

                            <div class="emotion">
                                {icon} {emotion}
                            </div>

                            <div class="confidence">
                                Confidence:
                                {confidence:.2f}%
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(
                        min(
                            confidence / 100,
                            1.0
                        )
                    )

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        🤖 <b>Emotion Detection using Deep Learning</b>
        <br><br>

        Developed by <b>Anuj Bhoir</b>
        <br>

        Python • TensorFlow • CNN • OpenCV • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)