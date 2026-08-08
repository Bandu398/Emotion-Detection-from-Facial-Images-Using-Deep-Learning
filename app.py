import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from pathlib import Path


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

.stApp {
    background-color: #f5f7fb;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #e8f0ff;
}

[data-testid="stSidebar"] * {
    color: #1f2937 !important;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #1e3a8a !important;
    margin-top: 15px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    font-size: 19px;
    color: #475569 !important;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background-color: #ffffff !important;
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #dbe3ef;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.08);
    margin-top: 20px;
}

.card * {
    color: #1f2937 !important;
}

.card h2 {
    color: #1e40af !important;
}

.card h3 {
    color: #2563eb !important;
}

.card p {
    color: #334155 !important;
    font-size: 17px;
    line-height: 1.6;
}

/* Feature */
.feature {
    background-color: #f8fafc;
    padding: 12px;
    margin: 8px 0;
    border-radius: 10px;
    color: #334155 !important;
    font-size: 17px;
}

/* Result */
.result-card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #dbe3ef;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.08);
    margin-top: 20px;
}

.result-card * {
    color: #1f2937 !important;
}

.result-title {
    color: #1e40af !important;
    font-size: 30px;
    font-weight: 700;
}

.confidence {
    color: #475569 !important;
    font-size: 18px;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 50px;
    padding: 20px;
    color: #64748b !important;
}

.footer b {
    color: #1e40af !important;
}

</style>
""", unsafe_allow_html=True)


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
            "❌ Model file not found! "
            "Please make sure model/emotion_model.h5 exists."
        )
        st.stop()

    return load_model(str(MODEL_PATH))


model = load_emotion_model()


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
# FACE DETECTOR
# =========================================================

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    st.error("❌ Face detector could not be loaded.")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/4712/4712027.png",
    width=120
)

st.sidebar.title("😊 Emotion AI")

st.sidebar.write("AI Facial Emotion Detection")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📂 Upload Image",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "AI / ML Internship Project 2026"
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.markdown(
        "<div class='main-title'>😊 Emotion Detection using AI</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sub-title'>"
        "Deep Learning • TensorFlow • OpenCV • Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

    # Metrics
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("🎭 Emotions", "7")

    with c2:
        st.metric("📂 Dataset", "FER2013")

    with c3:
        st.metric("🧠 Model", "CNN")


    # Project features

    st.markdown("""
    <div class="card">

    <h2>🚀 Project Features</h2>

    <div class="feature">
    ✅ Facial Emotion Detection
    </div>

    <div class="feature">
    ✅ CNN Deep Learning Model
    </div>

    <div class="feature">
    ✅ TensorFlow & Keras
    </div>

    <div class="feature">
    ✅ OpenCV Face Detection
    </div>

    <div class="feature">
    ✅ Streamlit Web Application
    </div>

    <div class="feature">
    ✅ FER2013 Dataset
    </div>

    <div class="feature">
    ✅ Confidence Score
    </div>

    </div>
    """, unsafe_allow_html=True)


    # How it works

    st.markdown("""
    <div class="card">

    <h2>⚙️ How It Works</h2>

    <p>
    <b>Step 1:</b> Upload a facial image.
    </p>

    <p>
    <b>Step 2:</b> OpenCV detects the face.
    </p>

    <p>
    <b>Step 3:</b> The face is resized to 48 × 48 pixels.
    </p>

    <p>
    <b>Step 4:</b> CNN model analyzes the facial features.
    </p>

    <p>
    <b>Step 5:</b> The system predicts the emotion.
    </p>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# UPLOAD IMAGE PAGE
# =========================================================

elif page == "📂 Upload Image":

    st.markdown(
        "<div class='main-title'>📂 Upload Face Image</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sub-title'>"
        "Upload an image and let AI detect the facial emotion"
        "</div>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )


    if uploaded_file is not None:

        # Open image
        image = Image.open(uploaded_file)

        # Convert RGB
        image = image.convert("RGB")

        # Convert to numpy
        img = np.array(image)


        # Display image

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )


        # Convert to grayscale

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )


        # Detect faces

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )


        if len(faces) == 0:

            st.error(
                "❌ No face detected. "
                "Please upload a clear face image."
            )


        else:

            st.success(
                f"✅ {len(faces)} face(s) detected!"
            )


            # Process each face

            for index, (x, y, w, h) in enumerate(faces):

                # Face ROI
                roi = gray[y:y+h, x:x+w]


                # Resize
                roi = cv2.resize(
                    roi,
                    (48, 48)
                )


                # Normalize
                roi = roi.astype("float32") / 255.0


                # Add channel dimension
                roi = np.expand_dims(
                    roi,
                    axis=-1
                )


                # Add batch dimension
                roi = np.expand_dims(
                    roi,
                    axis=0
                )


                # Prediction

                prediction = model.predict(
                    roi,
                    verbose=0
                )


                # Get emotion

                emotion_index = np.argmax(prediction)

                emotion = emotion_labels[
                    emotion_index
                ]


                # Confidence

                confidence = (
                    float(np.max(prediction)) * 100
                )


                # Result

                st.markdown(
                    f"""
                    <div class="result-card">

                    <div class="result-title">
                    😊 Face {index + 1}: {emotion}
                    </div>

                    <div class="confidence">
                    Confidence: {confidence:.2f}%
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.progress(
                    min(confidence / 100, 1.0)
                )


                # Emotion message

                if emotion == "Happy":
                    st.success(
                        "😄 The detected emotion is Happy!"
                    )

                elif emotion == "Sad":
                    st.info(
                        "😢 The detected emotion is Sad."
                    )

                elif emotion == "Angry":
                    st.warning(
                        "😠 The detected emotion is Angry."
                    )

                elif emotion == "Fear":
                    st.warning(
                        "😨 The detected emotion is Fear."
                    )

                elif emotion == "Disgust":
                    st.warning(
                        "🤢 The detected emotion is Disgust."
                    )

                elif emotion == "Surprise":
                    st.info(
                        "😲 The detected emotion is Surprise."
                    )

                else:
                    st.info(
                        "😐 The detected emotion is Neutral."
                    )


# =========================================================
# ABOUT PAGE
# =========================================================

else:

    st.markdown(
        "<div class='main-title'>ℹ️ About Project</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sub-title'>"
        "AI Based Facial Emotion Detection System"
        "</div>",
        unsafe_allow_html=True
    )


    st.markdown("""
    <div class="card">

    <h2>😊 Emotion Detection from Facial Images</h2>

    <p>
    This project is an Artificial Intelligence and Machine
    Learning application that detects human emotions from
    facial images using a trained CNN model.
    </p>


    <h3>🧠 Technologies Used</h3>

    <p>✅ Python</p>
    <p>✅ TensorFlow</p>
    <p>✅ Keras</p>
    <p>✅ OpenCV</p>
    <p>✅ NumPy</p>
    <p>✅ Streamlit</p>
    <p>✅ FER2013 Dataset</p>


    <h3>🎭 Emotions Detected</h3>

    <p>😠 Angry</p>
    <p>🤢 Disgust</p>
    <p>😨 Fear</p>
    <p>😄 Happy</p>
    <p>😐 Neutral</p>
    <p>😢 Sad</p>
    <p>😲 Surprise</p>


    <h3>🌐 Applications</h3>

    <p>• Education</p>
    <p>• Customer Feedback</p>
    <p>• Human Computer Interaction</p>
    <p>• Smart Applications</p>
    <p>• AI Based Monitoring</p>


    <h3>🎯 Project Objective</h3>

    <p>
    The main objective of this project is to use Artificial
    Intelligence and Deep Learning to automatically recognize
    human emotions from facial expressions.
    </p>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

<p>
Developed by <b>Anuj Bhoir</b>
</p>

<p>
AI/ML Internship Project 2026 🚀
</p>

</div>
""", unsafe_allow_html=True)
