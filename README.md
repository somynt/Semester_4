COCO Dataset: Focused Object Detection & Segmentation
Project Overview
This project demonstrates a complete machine learning workflow, from data analysis and preparation to model training and evaluation. The primary goal was to develop a highly efficient and accurate object detection and instance segmentation model tailored for a specific subset of the COCO dataset.

Methodology
The project's methodology was deliberately data-centric, focusing on creating an optimized and streamlined pipeline.

Exploratory Data Analysis (EDA): A detailed analysis of the raw COCO dataset was performed to understand key characteristics, including class distribution, object size, and spatial relationships. This initial step was crucial for identifying challenges such as class imbalance and for informing our model selection.

Dataset Filtering: The raw dataset was filtered to include only a predefined list of desired categories (e.g., person, car, dog). This strategic step significantly reduced the dataset size, leading to faster training and a more focused model. The coco_filter_and_analyze.py script was used for this process.

Model Selection & Training: A modern, single-stage detection model was chosen for its balance of high accuracy and computational efficiency. The model was trained on the filtered dataset with a comprehensive hyperparameter tuning process to achieve optimal performance.

Key Results
The final model's performance was rigorously validated, confirming the success of the methodology.

Overall Performance:

Box mAP50: 44.1%

Mask mAP50: 37.5%

Per-Class Performance: The model performed exceptionally well on high-frequency classes like person and car, with high precision and recall. Performance on low-frequency classes, such as cake, was understandably lower, highlighting the impact of class imbalance.

Efficiency: The model achieved an outstanding average inference speed of 0.9ms per image, making it highly suitable for real-time applications.

Future Directions
Future work will focus on further enhancing the model's performance and robustness. Potential areas for exploration include:

Applying advanced techniques (e.g., data augmentation or focal loss) to improve performance on low-frequency classes.

Exploring the integration of attention mechanisms to refine feature extraction.

Scaling the methodology to larger, more complex datasets.

Getting Started
To run the analysis yourself, ensure you have the required libraries installed. Place your COCO dataset in the specified paths and run the scripts in the following order:

enhanced_coco_eda.py

coco_filter_and_analyze.py

(Optional) Use the best.pt weights from your training run to perform inference on new images.
