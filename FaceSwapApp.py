import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from skimage.exposure import match_histograms

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def get_face_landmarks(image):
    try:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        return np.array([(lm.x * image.shape[1], lm.y * image.shape[0])
                         for lm in results.multi_face_landmarks[0].landmark])
    except:
        return None

def create_mask_from_landmarks(image_shape, landmarks):
    hull = cv2.convexHull(landmarks.astype(np.int32))
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    # Feather edges with Gaussian Blur
    mask = cv2.GaussianBlur(mask, (15, 15), 10)
    return mask

def apply_histogram_matching(source, target, mask):
    matched = match_histograms(source, target, channel_axis=-1)
    return np.where(mask[..., None] > 0, matched, target)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def robust_face_swap(img1, img2):
    img1_cv = np.array(img1.convert("RGB"))[:, :, ::-1]
    img2_cv = np.array(img2.convert("RGB"))[:, :, ::-1]

    lms1 = get_face_landmarks(img1_cv)
    lms2 = get_face_landmarks(img2_cv)
    if lms1 is None or lms2 is None or len(lms1) != len(lms2):
        raise ValueError("Could not detect valid faces in both images")

    # Use key feature points for better alignment
    key_indices = [33, 133, 362, 263, 1, 61, 291, 199]
    src_points = lms1[key_indices].astype(np.float32)
    dst_points = lms2[key_indices].astype(np.float32)

    # Estimate transform
    M, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
    if M is None:
        raise ValueError("Could not estimate alignment transform.")

    warped_face = cv2.warpAffine(img1_cv, M, (img2_cv.shape[1], img2_cv.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Create mask from landmarks
    mask = create_mask_from_landmarks(img2_cv.shape, lms2)

    # Histogram match for color consistency
    warped_face = apply_histogram_matching(warped_face, img2_cv, mask)

    # Blend with seamless clone
    center = tuple(np.mean(lms2, axis=0).astype(int))
    result = cv2.seamlessClone(warped_face, img2_cv, mask, center, cv2.NORMAL_CLONE)

    # Post-process: Sharpen result
    result = sharpen_image(result)

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# --- Streamlit UI ---
st.title("🔀 Swap Faces")

col1, col2 = st.columns(2)
with col1:
    img1 = st.file_uploader("Upload Face 1", type=["jpg", "jpeg", "png"])
with col2:
    img2 = st.file_uploader("Upload Face 2", type=["jpg", "jpeg", "png"])

if img1 and img2:
    img1 = Image.open(img1)
    img2 = Image.open(img2)

    st.subheader("Original Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption="Face 1", use_container_width=True)
    with col2:
        st.image(img2, caption="Face 2", use_container_width=True)

    if st.button("✨ Swap Faces"):
        with st.spinner("Swapping faces with improved accuracy..."):
            try:
                result = robust_face_swap(img1, img2)
                st.subheader("Swapped Result")
                st.image(result, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}\nTry clearer frontal face images")
