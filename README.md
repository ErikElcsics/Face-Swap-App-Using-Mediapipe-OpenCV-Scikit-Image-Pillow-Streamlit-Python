# 🔀 Face Swap App -- Using Mediapipe, OpenCV, Scikit-Image, Pillow, Streamlit, Python

This app uses **MediaPipe** and **OpenCV** to swap faces between two uploaded images. It combines advanced face landmark detection, image alignment, histogram matching, and seamless blending to produce high-quality face swaps. The app allows users to upload two images and instantly see the swapped faces in real-time.

## 🌟 Features

- 📸 **Face Detection & Landmarking**: Uses MediaPipe to detect key facial landmarks with high accuracy.
- 🔄 **Face Swap**: Swap faces between two images with seamless blending and improved quality.
- 🖼️ **Histogram Matching**: Ensures color consistency between the swapped face and target image.
- ✨ **Post-Processing**: Sharpening applied to the final image for better clarity.
- 🧑‍🤝‍🧑 **User-Friendly Interface**: Simple file upload for both images and intuitive face-swapping button.



## 🧠 Model Information

This app uses **MediaPipe Face Mesh**, an advanced model for detecting facial landmarks. It accurately identifies points on the face, which are then used to align and blend faces between two images.



## 🧩 How the Code Works - Summary

1. **Image Upload**: Users upload two images through the Streamlit interface.
2. **Face Detection**: MediaPipe Face Mesh detects facial landmarks in both images.
3. **Face Alignment**: The key landmarks of the two faces are matched to align the faces in both images.
4. **Face Swap**: The aligned face is warped onto the second image, and histogram matching is applied to maintain color consistency.
5. **Seamless Blending**: The swapped face is blended into the target image using OpenCV's seamless clone.
6. **Post-Processing**: The final image undergoes sharpening for clarity.


## 📦 Installation

**1. Clone the repository**

git clone https://github.com/ErikElcsics/Face-Swap-App-Using-Mediapipe-OpenCV-Scikit-Image-Pillow-Streamlit-Python.git
cd face-swap-app


**2. Create a virtual environment (optional but recommended)**

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


**3. Install dependencies**

pip install -r requirements.txt


**4. Run the app**

Launch the app using the command:
   
   streamlit run FaceSwapApp.py

## 🚀 How to Use the App
   
2. **Upload Two Images**: Use the file upload section to upload two face images.
3. **Face Swap**: After both images are uploaded, click the "✨ Swap Faces" button to process the swap.
4. **View Result**: The swapped image will appear below the original images.



## 🧰 Libraries Used

- [`streamlit`](https://streamlit.io/): Framework to create the web app interface.
- [`opencv-python`](https://opencv.org/): Library for image processing.
- [`mediapipe`](https://google.github.io/mediapipe/): Library for detecting face landmarks and processing images.
- [`numpy`](https://numpy.org/): Used for array manipulation.
- [`Pillow`](https://pillow.readthedocs.io/): Library to handle image operations.
- [`scikit-image`](https://scikit-image.org/): Used for histogram matching between images.



## 💡 App Summary

### From a User Perspective
- **Upload Images**: Upload two images with faces you want to swap.
- **Face Swap Button**: Click the button to process and swap the faces.
- **Result Display**: View the swapped face result instantly with high accuracy and color consistency.

### From a Technical Perspective
- **Face Landmark Detection**: The app uses MediaPipe Face Mesh to detect 468 facial landmarks on each face.
- **Image Alignment**: Key facial points are selected and used to calculate an affine transformation, aligning the faces.
- **Face Swap Process**:
  1. The first face is warped onto the second image using the affine transform.
  2. Histogram matching adjusts the colors of the warped face to match the target image.
  3. The faces are blended seamlessly using OpenCV’s `seamlessClone` function.
  4. A sharpening filter is applied to the final result for improved clarity.
- **Error Handling**: If faces are not detected or alignment fails, the app raises an error and requests clearer images.

![image](https://github.com/user-attachments/assets/71a1db2b-5128-44f8-af37-92b013ff01d6)

![image](https://github.com/user-attachments/assets/cd387785-119d-43c9-92ce-4bbfff83e604)



