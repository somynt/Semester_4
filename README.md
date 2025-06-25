# Semester_4
Assignment Submission_1
Hybrid Object Detection with Faster R-CNN, YOLOv8, and WBF
This repository contains Python code for a hybrid object detection system that combines the strengths of Faster R-CNN and YOLOv8 models using Weighted Box Fusion (WBF) for enhanced accuracy and robustness.

Table of Contents
Introduction
Features
Setup
Project Structure
Configuration
Usage
Debugging & Optimisation
Contributing
License
Introduction
This project implements a hybrid approach to object detection, integrating a fine-tuned Faster R-CNN model with a YOLOv8 model. By employing Weighted Box Fusion (WBF) as an ensemble method, the system aims to amalgamate bounding box predictions, leading to improved detection accuracy, enhanced recall, and increased resilience in challenging environmental conditions compared to individual standalone models. The system allows for analytical assessment and optimization of fusion parameters.

Features
Hybrid Model Integration: Combines Faster R-CNN (two-stage detector) and YOLOv8 (single-shot detector).
Weighted Box Fusion (WBF): Utilizes an advanced ensemble technique for robust prediction merging.
Configurable Thresholds: Adjustable confidence thresholds for individual models and the final WBF output.
IoU Control: Customizable Intersection-over-Union (IoU) threshold for the WBF process.
Visualization: Generates annotated images with detected bounding boxes and confidence scores.
Debugging Output: Provides detailed console output during inference to track detections at various stages.
Setup
Prerequisites
Python 3.x
PyTorch (and torchvision)
ultralytics (for YOLOv8)
ensemble-boxes
Pillow
matplotlib
numpy
opencv-python (though cv2 isn't directly used for core image loading in this specific code, it's a common dependency for vision tasks)
You can install the required Python packages using pip:

Bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # Use appropriate CUDA version or --cpu
pip install ultralytics ensemble-boxes Pillow matplotlib numpy
Model Checkpoints
You need to place your model checkpoints in the specified ROOT_DIR (e.g., /content/drive/MyDrive/ if running in Colab).

Faster R-CNN: fine_tuned_faster_rcnn_best.pth
YOLOv8: yolov8n.pt (or yolov8s.pt, yolov8m.pt if you downloaded a larger model)
Directory Structure (Example within your ROOT_DIR):

your_google_drive/
├── coco2017/
│   └── test-30/
│       └── 000000001439.jpg
│       └── ... (other test images)
├── hybrid_results/
│   └── (output images will be saved here)
├── fine_tuned_faster_rcnn_best.pth
├── yolov8n.pt
└── your_code_files/
    ├── model_setup.py          # <--- REPLACE WITH YOUR ACTUAL FILE NAMES
    ├── inference_functions.py  # <--- REPLACE WITH YOUR ACTUAL FILE NAMES
    └── main_hybrid_inference.py# <--- REPLACE WITH YOUR ACTUAL FILE NAMES
Project Structure
This project is likely organized into several logical components. Based on our conversation, the functionality might be distributed across these files:

model_setup.py (Hypothetical): Contains initial setup, device configuration, ROOT_DIR definition, CLASS_NAMES, and the model loading logic for Faster R-CNN and YOLOv8.
inference_functions.py (Hypothetical): Defines the faster_rcnn_inference, yolo_inference, and visualize_detections helper functions.
main_hybrid_inference.py (Hypothetical): Contains the perform_hybrid_inference_and_visualize function that orchestrates the entire process (calling inferences, WBF, final filtering), and the main execution block.
Please replace the hypothetical file names above with your actual file names.

Configuration
The HYBRID_CONFIG dictionary in your main script (main_hybrid_inference.py or similar) allows you to fine-tune the ensemble behavior:

Python

HYBRID_CONFIG = {
    'iou_threshold': 0.5,           # IoU threshold for NMS in WBF (how much boxes must overlap to be considered for merging)
    'conf_threshold_frcnn': 0.35,   # Minimum confidence for Faster R-CNN detections to be considered
    'conf_threshold_yolo': 0.25,    # Minimum confidence for YOLOv8 detections to be considered
    'weights': [0.5, 0.5],          # Weights assigned to Faster R-CNN and YOLOv8 outputs in WBF (sum should ideally be 1.0)
    'conf_threshold_wbf': 0.15      # Minimum confidence for a *fused* box to be a final detection
}
Usage
Place your models and test images in the appropriate directories as described in Setup.

Ensure your CLASS_NAMES list in the code matches the classes your models were trained on (with 'background' typically at index 0 for Faster R-CNN).

Update TEST_IMAGE_PATH in the main script to point to the image you want to test.

Run the main script:

Bash

python main_hybrid_inference.py # Replace with your actual main script file name
The script will print intermediate detection counts and scores, and save the visualized results to the hybrid_results directory within your ROOT_DIR.

Debugging & Optimisation
Console Output: The code includes extensive print statements to show the number of detections at each stage (raw model output, after individual thresholds, after WBF, after final WBF threshold). Use this to diagnose where detections are being filtered out.
Threshold Adjustment: Experiment with conf_threshold_frcnn, conf_threshold_yolo, and conf_threshold_wbf to balance precision and recall for your specific use case.
IoU Adjustment: For clustered objects, adjust iou_threshold in HYBRID_CONFIG. A higher value (e.g., 0.5-0.7) can help prevent merging of distinct, but overlapping, individuals.
Model Performance: If detections are consistently low confidence or missed, consider:
Fine-tuning your Faster R-CNN and YOLOv8 models on a dataset more representative of your challenging conditions (e.g., backlit, clustered, small objects).
Using larger YOLOv8 models (e.g., yolov8s.pt or yolov8m.pt) if yolov8n.pt is struggling with small objects.
Contributing
Feel free to open issues or submit pull requests.
