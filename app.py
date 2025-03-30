# import streamlit as st
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# import numpy as np
# import pandas as pd
# import cv2
# from pathlib import Path
# import sys
# import easyocr  # OCR library


# # Add YOLOv5 repository to Python path
# sys.path.append(str(Path.cwd() / "yolov5"))  # Update to your YOLOv5 directory

# # Import YOLOv5 utilities
# from yolov5.models.common import DetectMultiBackend
# from yolov5.utils.general import non_max_suppression, scale_boxes

# # Initialize session state for car database and new image processing flag
# if "car_database" not in st.session_state:
#     st.session_state.car_database = []  # Holds car data as a list of dictionaries
# if "is_new_image_processed" not in st.session_state:
#     st.session_state.is_new_image_processed = False  # Flag for new image processing
# if "use_camera" not in st.session_state:
#     st.session_state.use_camera = False  # To control when the camera is opened

# # Custom CSS for styling
# st.markdown("""
#     <style>
#         .main-title {
#             font-size: 40px;
#             font-weight: bold;
#             color: #4CAF50;
#             text-align: center;
#         }
#         .subheader {
#             color: #ff5722;
#             font-size: 22px;
#             font-weight: bold;
#         }
#         .highlight {
#             font-size: 20px;
#             font-weight: bold;
#             color: #673AB7;
#         }
#         .detected-text {
#             font-size: 18px;
#             color: #795548;
#             background-color: #f0f0f0;
#             padding: 5px;
#             border-radius: 5px;
#         }
#         .array-header {
#             font-size: 24px;
#             color: #3F51B5;
#             text-align: center;
#         }
#         .upload-container {
#             display: flex;
#             gap: 10px;
#             align-items: center;
#             justify-content: center;
#             margin-bottom: 20px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# @st.cache_resource
# def load_model():
#     model_path = "yolov5_model_windows.pt"  # Use the re-saved model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = DetectMultiBackend(model_path, device=device)
#     return model

# # Load the model
# model = load_model()

# # Image preprocessing function
# def preprocess_image(image):
#     transform = transforms.Compose([
#         transforms.Resize((640, 640)),  # Resize to 640x640
#         transforms.ToTensor(),
#     ])
#     return transform(image).unsqueeze(0)

# # Draw bounding boxes on the image and crop detected regions
# def draw_boxes_and_crop(image, detections, original_size):
#     image_np = np.array(image)
#     original_width, original_height = original_size
#     cropped_images = []
#     car_boxes = []

#     for det in detections:
#         x1, y1, x2, y2, conf, cls = det[:6]
#         box = torch.tensor([x1, y1, x2, y2]).unsqueeze(0)
#         scaled_box = scale_boxes((640, 640), box, (original_height, original_width))[0].tolist()
#         x1, y1, x2, y2 = map(int, scaled_box)

#         label = f"Class {int(cls)}, Conf {conf:.2f}"
#         cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red bounding box
#         cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#         cropped_image = image_np[y1:y2, x1:x2]
#         cropped_images.append(cropped_image)
#         car_boxes.append((x1, y1, x2, y2))

#     return image_np, cropped_images, car_boxes

# # OCR function using EasyOCR
# def extract_text_with_easyocr(cropped_images):
#     reader = easyocr.Reader(['en'])
#     extracted_texts = []
    
#     for cropped_image in cropped_images:
#         if cropped_image is not None and cropped_image.size > 0:
#             result = reader.readtext(cropped_image)
#             text = " ".join([res[1] for res in result])
#             extracted_texts.append(text)
#     return extracted_texts

# # Add car to the database or delete if it exists (only on new image processing)
# def update_car_database_on_new_image(car_number):
#     car_database = st.session_state.car_database
#     if st.session_state.is_new_image_processed:
#         existing_car = next((car for car in car_database if car["number"] == car_number), None)
#         if existing_car:
#             # Delete the existing car entry
#             st.session_state.car_database = [car for car in car_database if car["number"] != car_number]
#             st.warning(f"🗑️ Car with number '{car_number}' already exists. The entry has been removed from the database.")
#         else:
#             # Add a new car entry
#             car_id = f"{len(car_database) + 1:03}"
#             st.session_state.car_database.append({"id": car_id, "number": car_number})
#             st.success(f"🚗 New car '{car_number}' added to the system with ID: **{car_id}**.")
#         # Reset the flag after processing
#         st.session_state.is_new_image_processed = False

# # Display car database
# def display_car_database():
#     car_database = st.session_state.car_database
#     if car_database:
#         df = pd.DataFrame(car_database)
#         st.markdown('<h2 class="array-header">Current Car Database</h2>', unsafe_allow_html=True)
#         st.dataframe(df, use_container_width=True)
#     else:
#         st.info("🚫 No cars in the system.")

# # App layout
# st.markdown('<h1 class="main-title">YOLOv5 with OCR and Car Management</h1>', unsafe_allow_html=True)

# # Upload container with file uploader and camera button
# st.markdown('<div class="upload-container">', unsafe_allow_html=True)
# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
# if st.button("Take Photo with Camera"):
#     st.session_state.use_camera = True  # Enable camera capture
# st.markdown('</div>', unsafe_allow_html=True)

# # Camera capture (only if camera is enabled)
# camera_image = None
# if st.session_state.use_camera:
#     camera_image = st.camera_input("Capture an Image")

# # Determine the input source (uploaded image or camera image)
# if uploaded_file or camera_image:
#     if uploaded_file:
#         image = Image.open(uploaded_file).convert("RGB")
#     elif camera_image:
#         image = Image.open(camera_image).convert("RGB")
#         st.session_state.use_camera = False  # Reset camera flag

#     original_size = image.size

#     st.image(image, caption="📤 Uploaded Image", use_container_width=True)
#     st.markdown('<p class="highlight">Image uploaded successfully!</p>', unsafe_allow_html=True)

#     img_tensor = preprocess_image(image)

#     with st.spinner("🔍 Detecting objects..."):
#         pred = model(img_tensor, augment=False, visualize=False)
#         pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

#     st.markdown('<h2 class="subheader">Detection Results</h2>', unsafe_allow_html=True)
#     if pred[0] is not None and len(pred[0]) > 0:
#         image_with_boxes, cropped_images, car_boxes = draw_boxes_and_crop(image.copy(), pred[0], original_size)
#         st.image(image_with_boxes, caption="🔲 Detected Objects", use_container_width=True)

#         detected_texts = extract_text_with_easyocr(cropped_images)
#         for i, text in enumerate(detected_texts):
#             if text:
#                 st.session_state.is_new_image_processed = True  # Set flag when processing a new image
#                 update_car_database_on_new_image(text)
#                 st.markdown(f'<p class="detected-text">Detected Number Plate: {text}</p>', unsafe_allow_html=True)
#     else:
#         st.write("❌ No objects detected.")

# # Always display car database
# display_car_database()







import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import sys
import easyocr  # OCR library
import base64  # For encoding background image

# Add YOLOv5 repository to Python path
sys.path.append(str(Path.cwd() / "yolov5"))  # Update to your YOLOv5 directory

# Import YOLOv5 utilities
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes

# Initialize session state for car database and new image processing flag
if "car_database" not in st.session_state:
    st.session_state.car_database = []  # Holds car data as a list of dictionaries
if "is_new_image_processed" not in st.session_state:
    st.session_state.is_new_image_processed = False  # Flag for new image processing
if "use_camera" not in st.session_state:
    st.session_state.use_camera = False  # To control when the camera is opened

# Function to encode background image in base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Embed background image using CSS
background_image_path = r"C:\Users\hasee\Downloads\data of old laptop\7th semester\streamlit\back.jpeg"
# C:\Users\hasee\Downloads\data of old laptop\7th semester\streamlit\back.jpeg
background_image_base64 = get_base64_image(background_image_path)
st.markdown(
    f"""
    <style>
        body {{
            background-image: url("data:image/jpeg;base64,{background_image_base64}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .main-title {{
            font-size: 40px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }}
        .subheader {{
            color: #ff5722;
            font-size: 22px;
            font-weight: bold;
        }}
        .highlight {{
            font-size: 20px;
            font-weight: bold;
            color: #673AB7;
        }}
        .detected-text {{
            font-size: 18px;
            color: #795548;
            background-color: #f0f0f0;
            padding: 5px;
            border-radius: 5px;
        }}
        .array-header {{
            font-size: 24px;
            color: #3F51B5;
            text-align: center;
        }}
        .upload-container {{
            display: flex;
            gap: 10px;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }}
        .stFileUploader {{
            border-radius: 10px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.8);
        }}
        .stButton button {{
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 8px 16px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    model_path = "yolov5_model_windows.pt"  # Use the re-saved model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(model_path, device=device)
    return model

# Load the model
model = load_model()

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to 640x640
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Draw bounding boxes on the image and crop detected regions
def draw_boxes_and_crop(image, detections, original_size):
    image_np = np.array(image)
    original_width, original_height = original_size
    cropped_images = []
    car_boxes = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        box = torch.tensor([x1, y1, x2, y2]).unsqueeze(0)
        scaled_box = scale_boxes((640, 640), box, (original_height, original_width))[0].tolist()
        x1, y1, x2, y2 = map(int, scaled_box)

        label = f"Class {int(cls)}, Conf {conf:.2f}"
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red bounding box
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cropped_image = image_np[y1:y2, x1:x2]
        cropped_images.append(cropped_image)
        car_boxes.append((x1, y1, x2, y2))

    return image_np, cropped_images, car_boxes

# OCR function using EasyOCR
def extract_text_with_easyocr(cropped_images):
    reader = easyocr.Reader(['en'])
    extracted_texts = []
    
    for cropped_image in cropped_images:
        if cropped_image is not None and cropped_image.size > 0:
            result = reader.readtext(cropped_image)
            text = " ".join([res[1] for res in result])
            extracted_texts.append(text)
    return extracted_texts

# Add car to the database or delete if it exists (only on new image processing)
def update_car_database_on_new_image(car_number):
    car_database = st.session_state.car_database
    if st.session_state.is_new_image_processed:
        existing_car = next((car for car in car_database if car["number"] == car_number), None)
        if existing_car:
            # Delete the existing car entry
            st.session_state.car_database = [car for car in car_database if car["number"] != car_number]
            st.warning(f"🗑️ Car with number '{car_number}' already exists. The entry has been removed from the database.")
        else:
            # Add a new car entry
            car_id = f"{len(car_database) + 1:03}"
            st.session_state.car_database.append({"id": car_id, "number": car_number})
            st.success(f"🚗 New car '{car_number}' added to the system with ID: **{car_id}**.")
        # Reset the flag after processing
        st.session_state.is_new_image_processed = False

# Display car database
def display_car_database():
    car_database = st.session_state.car_database
    if car_database:
        df = pd.DataFrame(car_database)
        st.markdown('<h2 class="array-header">Current Car Database</h2>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("🚫 No cars in the system.")

# App layout
st.markdown('<h1 class="main-title">YOLOv5 with OCR and Car Management</h1>', unsafe_allow_html=True)

# Upload container with file uploader and camera button
st.markdown('<div class="upload-container">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if st.button("Take Photo with Camera"):
    st.session_state.use_camera = True  # Enable camera capture
st.markdown('</div>', unsafe_allow_html=True)

# Camera capture (only if camera is enabled)
camera_image = None
if st.session_state.use_camera:
    camera_image = st.camera_input("Capture an Image")

# Determine the input source (uploaded image or camera image)
if uploaded_file or camera_image:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
    elif camera_image:
        image = Image.open(camera_image).convert("RGB")
        st.session_state.use_camera = False  # Reset camera flag

    original_size = image.size

    st.image(image, caption="📤 Uploaded Image", use_container_width=True)
    st.markdown('<p class="highlight">Image uploaded successfully!</p>', unsafe_allow_html=True)

    img_tensor = preprocess_image(image)

    with st.spinner("🔍 Detecting objects..."):
        pred = model(img_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    st.markdown('<h2 class="subheader">Detection Results</h2>', unsafe_allow_html=True)
    if pred[0] is not None and len(pred[0]) > 0:
        image_with_boxes, cropped_images, car_boxes = draw_boxes_and_crop(image.copy(), pred[0], original_size)
        st.image(image_with_boxes, caption="🔲 Detected Objects", use_container_width=True)

        detected_texts = extract_text_with_easyocr(cropped_images)
        for i, text in enumerate(detected_texts):
            if text:
                st.session_state.is_new_image_processed = True  # Set flag when processing a new image
                update_car_database_on_new_image(text)
                st.markdown(f'<p class="detected-text">Detected Number Plate: {text}</p>', unsafe_allow_html=True)
    else:
        st.write("❌ No objects detected.")

# Always display car database
display_car_database()





