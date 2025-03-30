# YOLOv5 with OCR and Car Management

## Overview
This Streamlit-based web application leverages **YOLOv5** for object detection and **EasyOCR** for Optical Character Recognition (OCR) to extract text from detected regions. It is designed to detect and read car number plates, storing the extracted information in a database for easy management.

## Features
- 📷 **Upload Image or Capture from Camera**
- 🖼️ **Object Detection using YOLOv5**
- 🔍 **OCR using EasyOCR for Text Extraction**
- 📝 **Car Number Database Management** (Add/Delete Entries)
- 🎨 **Custom UI Styling with Streamlit & CSS**

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies
Make sure you have Python installed, then install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

## Project Structure
```
project-directory/
│-- app.py                 # Main Streamlit app
│-- yolov5/                # YOLOv5 model directory
│-- requirements.txt       # Python dependencies
│-- back.jpeg              # Background image
```

## Usage
1. Upload an image or capture from the camera.
2. The YOLOv5 model detects objects (e.g., car number plates).
3. EasyOCR extracts the number plate text.
4. The app updates the database with detected numbers.
5. View and manage stored entries directly within the app.

## Technologies Used
- **YOLOv5** for object detection
- **EasyOCR** for text recognition
- **Streamlit** for web interface
- **PyTorch** for model inference
- **OpenCV & PIL** for image processing

## Deployment
To deploy the app, follow these steps:



1. **Deploy on a Server (Local or Cloud)**
   ```bash
   streamlit run app.py --server.port 8501
   ```

## Contributing
Feel free to contribute by submitting issues or pull requests!

## License
MIT License. See `LICENSE` file for details.

