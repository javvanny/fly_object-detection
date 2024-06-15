import shutil
import io

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import uvicorn
import cv2
import os
import logging
import datetime
import torch

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from ultralytics import YOLO


# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

threshold_size_image = 720
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Словарь для отображения идентификаторов классов в названия объектов
object_classes = {0: "BPLA copter", 1: "airplane", 2: "helicopter", 3: "bird", 4: "BPLA airplane"}

yolo_model = YOLO('/app/yolov8m/weights/best.pt').to(device)

# Настройка путей для шаблонов и статических файлов
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Хранилище для загруженных файлов
UPLOAD_FOLDER = "./uploads"
RESULTS_FOLDER = "./results"

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Монтирование директории results как статической
app.mount("/results", StaticFiles(directory=RESULTS_FOLDER), name="results")

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model=yolo_model,
    confidence_threshold=0.62,
    device=device
)

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload_images/")
async def upload_images(files: List[UploadFile] = File(...)):
    uploaded_files = []
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            uploaded_files.append(file.filename)
        return {"message": "Images uploaded successfully", "files": uploaded_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")


@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Check if processed file already exists
        processed_file_path = os.path.join(RESULTS_FOLDER, f"{file.filename.split('.')[0]}_output.mp4")
        if os.path.exists(processed_file_path):
            return {"message": "Video already processed", "file": file.filename}

        return {"message": "Video uploaded successfully", "file": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")


@app.post("/detect_objects/")
async def detect_objects():
    try:
        results = []

        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            result_file_path = os.path.join(RESULTS_FOLDER, f"{filename.split('.')[0]}_result.txt")
            timeline_file_path = os.path.join(RESULTS_FOLDER, f"{filename.split('.')[0]}_timeline.txt")
            try:
                if file_path.endswith(('.jpg', '.jpeg', '.png')):
                    # Process image files
                    image = cv2.imread(file_path)
                    detections = process_image(image)
                    write_detections(result_file_path, detections, image.shape)
                    results.append(result_file_path)

                elif file_path.endswith(('.mp4', '.avi')):
                    # Process video files
                    output_file_path = process_video(file_path, result_file_path, timeline_file_path)
                    results.append(output_file_path)

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

        return {"message": "Detection completed", "results": results}
    except Exception as e:
        print(e)


def process_image(image):
    image_height, image_width = image.shape[:2]

    if image_height < threshold_size_image or image_width < threshold_size_image:
        detections = yolo_model(image)
    else:
        detections = windowed_detection(image, detection_model)
    return detections


def write_detections(file_path, detections, image_shape):
    try:
        image_height, image_width = image_shape[:2]

        with open(file_path, "w") as f:
            if image_height < threshold_size_image or image_width < threshold_size_image:
                for detection in detections:
                    for box in detection.boxes.xywhn:
                        x_center, y_center, width, height = box.tolist()
                        class_id = int(detection[0].boxes.cls.cpu().item())
                        f.write(f"{class_id};{x_center};{y_center};{width};{height}\n")
            else:
                for detection in detections:
                    f.write(f"{detection}\n")
    except Exception as e:
        print(e)


def process_video(file_path, result_file_path, timeline_file_path):
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail=f"Unable to open video file {file_path}")

        output_file_path = os.path.join(RESULTS_FOLDER, f"{file_path.split('.')[0]}result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        with open(result_file_path, "w") as f, open(timeline_file_path, "w") as tlf:
            frame_number = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_height, frame_width = frame.shape[:2]
                detections = process_frame(frame, frame_height, frame_width)

                objects_detected = False
                detected_objects = set()

                for detection in detections:
                    if frame_height < threshold_size_image or frame_width < threshold_size_image:
                        for box in detection.boxes.xywhn:
                            x_center, y_center, width, height = box.tolist()
                            class_id = list(detection.names.keys())[0]
                            f.write(f"{class_id};{x_center};{y_center};{width};{height}\n")
                            objects_detected = True
                            detected_objects.add(object_classes.get(int(class_id), "Неизвестный объект"))
                    else:
                        class_id, x_center, y_center, width, height = detection.split(';')
                        f.write(f"{class_id};{x_center};{y_center};{width};{height}\n")
                        objects_detected = True
                        detected_objects.add(object_classes.get(int(class_id), "Неизвестный объект"))

                if objects_detected:
                    timestamp = str(datetime.timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)))
                    detected_objects_str = ", ".join(detected_objects)
                    tlf.write(f"{timestamp}: {detected_objects_str}\n")

                    # Draw bounding boxes on the frame
                    if frame_height < threshold_size_image or frame_width < threshold_size_image:
                        frame = visualize_detections(frame, [
                            (list(detection.names.keys())[0], x_center, y_center, width, height)
                            for detection in detections
                            for box in detection.boxes.xywhn
                        ])
                    else:
                        frame = visualize_detections(frame, [
                            (class_id, float(x_center), float(y_center), float(width), float(height))
                            for detection in detections
                        ])

                out.write(frame)
                frame_number += 1

        cap.release()
        out.release()
        return output_file_path
    except Exception as e:
        print(e)


def process_frame(frame, frame_height, frame_width):
    if frame_height < threshold_size_image or frame_width < threshold_size_image:
        detections = yolo_model(frame)
    else:
        detections = windowed_detection(frame, detection_model)
    return detections


def windowed_detection(image, detection_model):
    try:
        detections = get_sliced_prediction(image, detection_model,
                                           slice_height=640,
                                           slice_width=640,
                                           overlap_height_ratio=0.2,
                                           overlap_width_ratio=0.2)

        detections = detections.to_coco_annotations()
        res = convert_to_normalized_format(detections, image)
        print(res)
    except Exception as e:
        print(e)
        res = []

    return res


def normalize_bbox(bbox, image_width, image_height):
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height
    return x_center, y_center, width, height


def convert_to_normalized_format(detections, image):
    image_height, image_width = image.shape[:2]
    normalized_data = []

    for detection in detections:
        category_id = detection['category_id']
        bbox = detection['bbox']
        x_center, y_center, width, height = normalize_bbox(bbox, image_width, image_height)
        normalized_data.append(f"{category_id};{x_center};{y_center};{width};{height}")

    return normalized_data


def visualize_detections(frame, detections):
    for class_id, x_center, y_center, width, height in detections:
        x_center = int(x_center * frame.shape[1])
        y_center = int(y_center * frame.shape[0])
        width = int(width * frame.shape[1])
        height = int(height * frame.shape[0])
        x1 = x_center - width // 2
        y1 = y_center - height // 2
        x2 = x_center + width // 2
        y2 = y_center + height // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame


@app.get("/get_timeline/")
async def get_timeline(file: str):
    timeline_file_path = os.path.join(RESULTS_FOLDER, f"{file.split('.')[0]}_timeline.txt")
    if not os.path.exists(timeline_file_path):
        raise HTTPException(status_code=404, detail="Timeline file not found")

    with open(timeline_file_path, "r") as f:
        timestamps = f.readlines()

    return {"timeline": [ts.strip() for ts in timestamps]}


@app.get("/play_video/")
async def play_video(file: str):
    video_file_path = os.path.join(RESULTS_FOLDER, f"{file.split('.')[0]}_output.mp4")
    if not os.path.exists(video_file_path):
        raise HTTPException(status_code=404, detail="Processed video file not found")

    return StreamingResponse(open(video_file_path, "rb"), media_type="video/mp4")


@app.post("/show_detection/")
async def show_detection(file: UploadFile = File(...)):
    file_path = os.path.join(RESULTS_FOLDER, file.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    image = cv2.imread(file_path)
    result_file_path = os.path.join(RESULTS_FOLDER, f"{file.filename.split('.')[0]}.txt")
    detections = []
    try:
        with open(result_file_path, "r") as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split(";"))
                detections.append((class_id, x_center, y_center, width, height))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read detection results: {e}")

    image = visualize_detections(image, detections)
    _, im_png = cv2.imencode(".png", image)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


def visualize_detections(image, detections):
    for det in detections:
        class_id, x_center, y_center, width, height = det
        x_center = int(x_center * image.shape[1])
        y_center = int(y_center * image.shape[0])
        width = int(width * image.shape[1])
        height = int(height * image.shape[0])
        x1 = x_center - width // 2
        y1 = y_center - height // 2
        x2 = x_center + width // 2
        y2 = y_center + height // 2
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


@app.get("/download_results/")
async def download_results():
    shutil.make_archive('results', 'zip', RESULTS_FOLDER)
    return FileResponse('results.zip', media_type='application/x-zip-compressed', filename='results.zip')


@app.get("/show_results/")
async def show_results(request: Request):
    result_files = [f for f in os.listdir(RESULTS_FOLDER) if os.path.isfile(os.path.join(RESULTS_FOLDER, f))]
    return templates.TemplateResponse("results.html", {"request": request, "result_files": result_files})


@app.post("/clear_folders/")
async def clear_folders():
    for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
    return {"message": "Folders cleared successfully"}


@app.get("/progress/")
async def get_progress():
    global progress
    return {"progress": progress}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
