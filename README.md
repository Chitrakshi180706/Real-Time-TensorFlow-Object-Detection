# Real-Time Object Detection Using TensorFlow & OpenCV

An end-to-end computer vision application that performs real-time multi-class object detection on live webcam feeds using TensorFlow, the TensorFlow Object Detection API, and OpenCV.

Designed for college project submissions, technical portfolio demonstrations, and GitHub/LinkedIn project showcases.

---

## 📌 Overview

This project implements a real-time object detection pipeline that captures frames from a webcam feed, processes them through a pre-trained **Single Shot MultiBox Detector (SSD) with MobileNet V1** architecture trained on the COCO dataset, and displays the annotated video stream with bounding boxes, object category labels, confidence percentages, and a live FPS indicator.

---

## ✨ Features

- **Live Webcam Object Detection**: Detects 90+ common object categories in real-time (e.g., person, bottle, laptop, cell phone, chair, dog, cat, car).
- **Robust Windows Camera Backend Auto-Detection**: Automatically tests and switches between `DirectShow (CAP_DSHOW)`, `Media Foundation (CAP_MSMF)`, and default OpenCV backends across indices `0, 1, 2` to prevent black webcam screens.
- **Bounding Box & Label Overlay**: Draws high-accuracy bounding boxes with class names and confidence percentages (e.g., `Person 95%`).
- **Real-Time FPS Performance Overlay**: Calculates and displays smoothed frames-per-second performance metrics on the video stream.
- **Modular Test Suite**: Includes standalone test scripts for verifying camera feeds (`camera_test.py`), model inference (`model_test.py`), and environment configuration (`setup.py`).
- **Cross-Version TensorFlow 2.x Compatibility**: Backwards-compatible utility wrappers ensuring seamless operation on modern TensorFlow 2.x releases.

---

## 🛠️ Technologies Used

- **Python 3.7.16**
- **TensorFlow 2.3.0** (CPU optimized execution)
- **OpenCV (opencv-python 5.0.0 / 4.5+)**
- **TensorFlow Object Detection API**
- **Google Protobuf 3.20.3**
- **NumPy 1.18.5**
- **Matplotlib & Pillow**

---

## 📁 Project Structure

```
Real_time_object_detection_using_tensorflow/
│
├── real_time_object_detection.py   # Main real-time object detection application
├── camera_test.py                  # Standalone webcam diagnostic test script
├── model_test.py                   # Standalone model loading & inference test script
├── setup.py                        # Full environment setup check script
├── requirements.txt                # Pinned dependency requirements
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
│
├── ssd_mobilenet_v1_coco_2017_11_17/
│   └── frozen_inference_graph.pb   # Pretrained model frozen inference graph
│
├── models/
│   └── research/
│       └── object_detection/       # TensorFlow Object Detection API package
│           ├── data/
│           │   └── mscoco_label_map.pbtxt # Category label map definition
│           └── protos/             # Protobuf definitions and compiled *_pb2.py modules
└── object_detection_tutorial_Webcam_1.ipynb # Interactive Jupyter Notebook
```

---

## 💻 Environment Setup & Installation

### 1. Activate Environment

Open your Conda Command Prompt or Terminal and activate the project environment:

```bash
conda activate object_detection
```

### 2. Install Dependencies

Install the pinned compatible dependency suite:

```bash
pip install -r requirements.txt
```

---

## 🧪 Verification & Testing

Before launching the main application, run the automated verification scripts to validate your setup.

### Step 1: Run Full Setup Check

Verify Python packages, TensorFlow 2.x compatibility, proto bindings, and model files:

```bash
python setup.py
```

*Expected Output:*
```
========================================
SETUP CHECK PASSED
========================================
```

### Step 2: Run Model Inference Test

Verify graph parsing, label map loading, and model inference independently of the camera:

```bash
python model_test.py
```

*Expected Output:*
```
========================================
MODEL TEST PASSED
========================================
```

### Step 3: Run Camera Diagnostic Test

Test webcam connection, Windows backend selection, resolution, and frame brightness:

```bash
python camera_test.py
```

*Expected Window:* A window titled `Camera Test Feed` displaying your live camera image with diagnostic telemetry (Backend, Index, Resolution, Mean Brightness, FPS). Press **`q`** to exit.

---

## 🚀 Running the Main Application

To start live real-time object detection:

```bash
python real_time_object_detection.py
```

### Optional Command Line Arguments

- **Specify Camera Index** (e.g., external USB camera at index 1):
  ```bash
  python real_time_object_detection.py --camera-index 1
  ```
- **Set Minimum Confidence Threshold** (e.g., show detections above 60% confidence):
  ```bash
  python real_time_object_detection.py --min-score 0.6
  ```
- **Record Video Output**:
  ```bash
  python real_time_object_detection.py --save-video --output-path detection_output.avi
  ```

---

## 🎮 Controls

| Action | Control |
| :--- | :--- |
| **Exit Application** | Press **`q`** while focusing on the camera feed window |
| **Close Camera Window** | Press **`q`** or `Ctrl + C` in terminal |

---

## 🔍 Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **Black Webcam Window** | Run `python camera_test.py`. Ensure physical camera privacy shutter is open and Windows Camera Privacy permissions are granted (`Settings -> Privacy & Security -> Camera`). |
| **`ModuleNotFoundError: No module named 'protos'`** | Ensure `setup.py` has been run. Project scripts automatically append `models/research` to `sys.path`. |
| **`AttributeError: module 'tensorflow' has no attribute 'gfile'`** | Project utilities include automatic alias `tf.gfile = tf.io.gfile` for TensorFlow 2.x compatibility. |
| **Low FPS / Lag** | MobileNet V1 is optimized for CPU inference (~10-20 FPS on modern CPUs). Close background applications to free up CPU resources. |

---

## 🔮 Future Improvements

- Integrate lightweight YOLOv8 / MediaPipe models for higher CPU frame rates.
- Add GPU acceleration support via TensorRT or CUDA.
- Implement object counting and tracking (e.g., DeepSORT) across frames.
