# Drone Detection with YOLOv8

This project provides a FastAPI-based application for detecting drones and other objects in images and videos using the YOLOv8 model. The application supports uploading images and videos, running object detection, and retrieving the results.

## Features

- Upload and process images and videos for object detection.
- View detection results and download them.
- Display detection timelines for videos.
- Clear uploaded files and results.
- Show progress of detection processing.

## Requirements

- Python 3.8
- FastAPI
- Uvicorn
- OpenCV
- Torch
- Sahi
- Ultralytics YOLO

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/drone-detection.git
    cd drone-detection
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

3. Download the YOLOv8 model weights and place them in the specified directory:

    ```sh
    # Ensure the weights file is located at /app/yolov8m/weights/best.pt
    ```

## Usage

1. Start the FastAPI server:

    ```sh
    uvicorn main:app --reload
    ```

2. Open your browser and navigate to `http://localhost:8000` to access the web interface.

## Endpoints

### GET `/`

Renders the main page with options to upload images or videos.

### POST `/upload_images/`

Uploads images for object detection.

- **Parameters:**
  - `files`: List of image files to upload.

### POST `/upload_video/`

Uploads a video for object detection.

- **Parameters:**
  - `file`: Video file to upload.

### POST `/detect_objects/`

Runs object detection on the uploaded files.

### GET `/get_timeline/`

Retrieves the detection timeline for a specified video.

- **Parameters:**
  - `file`: The name of the video file.

### GET `/download_results/`

Downloads a zip file containing all detection results.

### GET `/show_results/`

Shows a list of result files available for download.

### POST `/clear_folders/`

Clears all uploaded files and results.

### GET `/progress/`

Shows the progress of detection processing.

## Example

1. Upload an image or video using the web interface.
2. Click the "Detect Objects" button to run the detection.
3. Downloads a zip file containing all detection results.

## Project Structure
├── main.py

├── requirements.txt

├── templates

   ├── index.html

    └── results.html
├── static
  
   └── ...

├── uploads

├── results

└── yolov8m
  
  └── weights
   
    └── best.pt
## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For any inquiries, please contact cap_emela@mail.ru.
