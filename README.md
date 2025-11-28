# Seatbelt Detection System

A computer vision application that uses a YOLOv11 model to detect seatbelt usage in video footage. This system processes video input, detects drivers/passengers, and classifies them based on whether they are wearing a seatbelt.

## Features

- **Real-time Detection**: Uses YOLOv11 for fast and accurate object detection.
- **Visual Feedback**:
  - **Green Bounding Box**: Seatbelt detected.
  - **Red Bounding Box**: No seatbelt detected.
- **Video Output**: Saves the processed video with annotations to the `output/` directory.
- **Configurable**: Easy to adjust detection thresholds and input sources.

## Prerequisites

- Python 3.8 or higher
- A CUDA-capable GPU is recommended for faster processing (though not strictly required, it significantly improves performance).

## Installation

1.  **Clone the repository** (or download the source code):
    ```bash
    git clone <repository-url>
    cd "Seatbelt Detection"
    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Prepare your environment**:
    Ensure you have the YOLO model file (`best.pt`) in the `models/` directory and an input video in `sample/`.

2.  **Run the detection script**:
    ```bash
    python main.py
    ```

3.  **Controls**:
    - The processed video will be displayed in a window.
    - Press `q` to stop the processing early.

4.  **Output**:
    - The processed video will be saved in the `output/` folder with a timestamped filename (e.g., `test_result_20231027120000.mp4`).

## Configuration

You can modify the following variables in `main.py` to customize the behavior:

- `MODEL_PATH`: Path to your trained YOLOv11 model (`.pt` file).
- `INPUT_VIDEO`: Path to the input video file.
- `THRESHOLD_SCORE`: Confidence threshold for detections (default: `0.5`).
- `MAX_FRAME_RECORD`: Maximum number of frames to process (default: `500`). Set to a larger number or remove the limit for full video processing.
- `SKIP_FRAMES`: Process every Nth frame to speed up processing (default: `1`).

## Project Structure

```
Seatbelt Detection/
├── main.py             # Main application script
├── requirements.txt    # Python dependencies
├── models/             # Directory for YOLO models
│   └── best.pt         # Trained model file
├── output/             # Directory for saved results
├── sample/             # Directory for input videos
└── README.md           # Project documentation
```

## License

[Choose a license, e.g., MIT, Apache 2.0, or leave blank if private]
