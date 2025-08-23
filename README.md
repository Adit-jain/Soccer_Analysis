# Soccer Analysis Project

A comprehensive soccer analysis system that implements computer vision for tracking players, ball, and referees in soccer videos. The system uses YOLO for object detection, ByteTrack for multi-object tracking, and SigLIP embeddings with UMAP/K-means clustering for team assignment.

## 🏗️ Architecture Overview

The project follows a **modular architecture** with strict separation of concerns:
- **Independent Core Modules**: No cross-dependencies between modules
- **Pipeline-Based Coordination**: Pipelines orchestrate module interactions
- **Clean Separation**: Detection core vs. pipeline functions separated
- **Crops-Only Clustering**: Team assignment works on pre-extracted player crops

## 📁 Project Structure

```
Soccer_Analysis/
├── README.md                        # This file
├── CLAUDE.md                        # Developer documentation
├── main.py                          # Main entry point
├── constants.py                     # Global configuration
├── 
├── 🔧 Core Modules (Independent)
├── player_detection/                # YOLO detection functions
│   ├── __init__.py
│   ├── detect_players.py            # Core detection logic
│   ├── detection_constants.py       # Detection constants
│   └── training/                    # Model training
│       ├── config.py
│       ├── main.py
│       └── trainer.py
├── player_tracking/                 # ByteTrack tracking
│   ├── __init__.py
│   └── tracking.py                  # TrackerManager
├── player_annotations/              # Visualization
│   ├── __init__.py
│   └── annotators.py                # AnnotatorManager
├── player_clustering/               # Team assignment
│   ├── __init__.py
│   ├── embeddings.py                # SigLIP embeddings
│   └── clustering.py                # UMAP + K-means
├── 
├── 🚰 Pipeline Layer (Coordination)
├── pipelines/                       # Module coordination
│   ├── __init__.py
│   ├── tracking_pipeline.py         # Complete tracking pipeline
│   ├── detection_pipeline.py        # Detection workflows
│   └── processing_pipeline.py       # Video I/O utilities
├── 
├── 🛠️ Utilities & Data
├── utils/                           # General utilities
│   ├── __init__.py
│   └── vid_utils.py                 # Video I/O functions
├── Data_utils/                      # Dataset preparation
│   ├── External_Detections/         # COCO/YOLO utilities
│   ├── SoccerNet_Detections/        # SoccerNet detection data
│   └── SoccerNet_Keypoints/         # Field keypoint processing
├── keypoint_detection/              # Field keypoint detection
├── 
├── 📦 Models & Data
├── Models/
│   ├── Pretrained/                  # Base YOLO models
│   └── Trained/                     # Fine-tuned models
└── yolo11n.pt                       # Base model file
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Soccer_Analysis
```

### 2. Install Dependencies
```bash
pip install ultralytics supervision torch torchvision transformers scikit-learn umap-learn pandas numpy opencv-python tqdm more-itertools pillow
```

### 3. Download the Trained Model

The project uses a custom-trained YOLO model available on Hugging Face:

**Model URL**: https://huggingface.co/Adit-jain/soccana

#### Option A: Manual Download
1. Visit https://huggingface.co/Adit-jain/soccana
2. Download the model file (typically `best.pt` or similar)
3. Place it in the appropriate directory structure

#### Option B: Using Hugging Face Hub (Recommended)
```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download the model programmatically
python -c "
from huggingface_hub import hf_hub_download
import os

# Download model
model_file = hf_hub_download(
    repo_id='Adit-jain/soccana',
    filename='best.pt'  # Adjust filename as needed
)

# Create directory structure
os.makedirs('Models/Trained/yolov11_sahi_1280/First/weights', exist_ok=True)

# Move model to expected location
import shutil
shutil.copy(model_file, 'Models/Trained/yolov11_sahi_1280/First/weights/best.pt')
print('Model downloaded and placed successfully!')
"
```

### 4. Update Configuration

Edit `constants.py` to set up your paths:

```python
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

# Model path (update if you placed model elsewhere)
model_path = r"Models\Trained\yolov11_sahi_1280\First\weights\best.pt"
model_path = PROJECT_DIR / model_path

# Video paths (update with your test video paths)
test_video = r"path\to\your\test_video.mp4"           # Input video
test_video_output = r"path\to\output\tracked_video.mp4" # Output video
```

### 5. Run the Complete Pipeline
```bash
python main.py
```

## 🎯 Usage Examples

### Basic Usage - Complete Pipeline
```python
from main import track_players
from constants import model_path, test_video

# Run complete tracking pipeline with team assignment
track_players(test_video, model_path)
```

### Advanced Usage - Custom Pipelines

#### Detection Only
```python
from pipelines import DetectionPipeline

# Initialize detection pipeline
detection_pipeline = DetectionPipeline(model_path)

# Run detection on video
detection_pipeline.detect_in_video("input.mp4", "output_detected.mp4")

# Real-time detection
detection_pipeline.detect_realtime("input.mp4")  # or 0 for webcam
```

#### Custom Tracking Pipeline
```python
from pipelines import TrackingPipeline, ProcessingPipeline

# Initialize pipelines
tracking_pipeline = TrackingPipeline(model_path)
processing_pipeline = ProcessingPipeline()

# Initialize models
tracking_pipeline.initialize_models()

# Train team assignment (if needed)
tracking_pipeline.train_team_assignment_models(video_path)

# Process video
frames = processing_pipeline.read_video_frames(video_path)
tracks = tracking_pipeline.get_tracks(frames)
tracks = processing_pipeline.interpolate_ball_tracks(tracks)
annotated_frames = tracking_pipeline.annotate_frames(frames, tracks)

# Save output
output_path = processing_pipeline.generate_output_path(video_path)
processing_pipeline.write_video_output(annotated_frames, output_path)
```

#### Using Individual Modules (Independent)
```python
# Each module can be used independently
from player_tracking import TrackerManager
from player_clustering import ClusteringManager
from player_annotations import AnnotatorManager
from player_detection import load_detection_model, get_detections

# Initialize components
tracker = TrackerManager()
clustering = ClusteringManager()
annotator = AnnotatorManager()
model = load_detection_model(model_path)

# Use individually
# ... custom logic combining modules
```

## 🔄 System Workflow

### Complete Tracking Pipeline Flow

1. **Model Initialization**
   - Load YOLO detection model
   - Initialize ByteTracker
   - Initialize SigLIP embedding extractor
   - Initialize UMAP and K-means models

2. **Team Assignment Training** (First-time setup)
   - Extract video frames (stride=12, first 120*24 frames)
   - Detect players in frames
   - Extract player crops from detections
   - Generate SigLIP embeddings from crops
   - Train UMAP dimensionality reduction
   - Train K-means clustering (k=2 for two teams)

3. **Video Processing**
   - For each frame:
     - **Detection**: YOLO detects players, ball, referees
     - **Tracking**: ByteTrack assigns consistent IDs
     - **Team Assignment**: Crop players → embeddings → clustering
     - **Storage**: Store tracks with IDs and team assignments

4. **Post-Processing**
   - **Ball Interpolation**: Fill missing ball detections
   - **Annotation**: Draw bounding boxes, IDs, team colors
   - **Output**: Generate tracked video

### Module Independence

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ player_detection│    │ player_tracking │    │player_annotations│
│                 │    │                 │    │                 │
│ • YOLO models   │    │ • ByteTrack     │    │ • Visualizations│
│ • get_detections│    │ • TrackerManager│    │ • AnnotatorMgr  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                 ▲
                                 │
                    ┌─────────────────┐
                    │player_clustering│
                    │                 │
                    │ • SigLIP embeds │
                    │ • UMAP + K-means│
                    │ • Crops-only    │
                    └─────────────────┘
                                 ▲
                                 │
              ┌──────────────────────────────────────┐
              │            pipelines/                │
              │                                      │
              │  • Coordinates all modules          │
              │  • No module talks to another       │
              │  • Handles data flow & crops        │
              └──────────────────────────────────────┘
```

## 🎮 Detection Classes

The system detects three main object classes:
- **Class 0**: Players
- **Class 1**: Ball
- **Class 2**: Referee

## ⚙️ Configuration

### Global Configuration (`constants.py`)
```python
# Model configuration
model_path = PROJECT_DIR / "Models/Trained/yolov11_sahi_1280/First/weights/best.pt"

# Video paths
test_video = "path/to/input/video.mp4"
test_video_output = "path/to/output/video.mp4"
```

### Detection Configuration (`player_detection/detection_constants.py`)
- Detection-specific parameters
- Model thresholds
- Class mappings

### Training Configuration (`player_detection/training/config.py`)
- Training hyperparameters
- Dataset paths
- Augmentation settings

## 🏃‍♂️ Performance Features

- **Batch Processing**: SigLIP embeddings processed in batches (batch_size=24)
- **GPU Acceleration**: Automatic GPU usage for PyTorch models
- **SAHI Support**: Slicing Aided Hyper Inference for large images
- **Ball Interpolation**: Linear interpolation with 30-frame limit
- **Modular Optimization**: Each module can be optimized independently

## 🧪 Testing

The modular architecture makes testing straightforward:

```python
# Test individual modules
from player_tracking import TrackerManager
from player_clustering import ClusteringManager

# Test tracking independently
def test_tracking():
    tracker = TrackerManager()
    # ... test tracking logic

# Test clustering with pre-prepared crops
def test_clustering():
    clustering = ClusteringManager()
    # ... test with crops data

# Test pipelines
from pipelines import TrackingPipeline
def test_pipeline():
    pipeline = TrackingPipeline(model_path)
    # ... test pipeline logic
```

## 🔧 Development

### Adding New Modules
1. Create module in base directory
2. Ensure no dependencies on other custom modules
3. Add pipeline coordination if needed
4. Update imports in relevant pipelines

### Creating Custom Pipelines
1. Create new file in `pipelines/`
2. Import required modules
3. Implement coordination logic
4. Maintain module independence

### Model Training
```bash
# Configure training parameters
vim player_detection/training/config.py

# Run training
python player_detection/training/main.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Not Found**
   ```
   FileNotFoundError: Model file not found
   ```
   **Solution**: Check `constants.py` model path and ensure model is downloaded

2. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce batch size in clustering configuration or use CPU

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'player_tracking'
   ```
   **Solution**: Ensure you're running from project root directory

4. **Video File Issues**
   ```
   cv2.error: Video file cannot be opened
   ```
   **Solution**: Check video file path and format compatibility

### Debug Mode
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug info
from pipelines import TrackingPipeline
pipeline = TrackingPipeline(model_path)
# ... pipeline will show detailed logs
```

## 📊 SoccerNet Data Processing

The system includes specialized tools for SoccerNet dataset:

### Keypoint Processing
```bash
# Extract field keypoints from line data
python Data_utils/SoccerNet_Keypoints/get_kepoints_from_lines.py

# Detect pitch objects for training
python Data_utils/SoccerNet_Keypoints/get_pitch_object.py
```

### Detection Data
```bash
# Process SoccerNet detection annotations
python Data_utils/SoccerNet_Detections/data_preprocessing.py
```

---

**Model URL**: https://huggingface.co/Adit-jain/soccana

Make sure to download the trained model and update `constants.py` before running the system!