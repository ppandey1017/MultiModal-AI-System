# MultiModal-AI-System

An AI-powered multi-modal classification system built using TensorFlow, NLP, CNN, and YOLO for email spam detection and image object detection.

This project combines Natural Language Processing and Computer Vision into one unified AI automation pipeline.

---

## Features

* Email Spam Detection using NLP & TensorFlow
* TF-IDF Vectorization & Advanced Text Preprocessing
* Real-World Large-Scale Spam Dataset Training
* Dataset Balancing and Duplicate Removal
* Confusion Matrix & Performance Visualization
* Accuracy & Loss Graph Generation
* YOLOv8 Object Detection for Multiple Images
* Automated Image + Text Classification Pipeline
* Saved Trained Models for Reusability

---

## Technologies Used

* Python
* TensorFlow / Keras
* NLP (Natural Language Processing)
* TF-IDF Vectorization
* CNN (Convolutional Neural Networks)
* YOLOv8
* OpenCV
* Scikit-learn
* Matplotlib
* NLTK

---

## Model Performance

* Trained on a real-world spam dataset containing 5000+ email samples
* Train/Test Split implemented
* Dataset Balancing applied for better prediction fairness
* Accuracy Evaluation added
* Confusion Matrix generated
* Training Accuracy & Loss Graphs visualized
* NLP preprocessing includes:

  * Stopword Removal
  * Text Cleaning
  * Lowercase Conversion
  * Stemming

---

## Object Detection

YOLOv8 is integrated for real-time image object detection.

Example detections:

* Dog
* Cat
* Car
* Person
* Laptop
* Chair

Detected output images are automatically generated after prediction.

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the project

```bash
python main.py
```

---

## Project Structure

```text
MultiModal-AI-System/
│
├── main.py
├── text_classifier.py
├── yolo_detector.py
├── cnn_classifier.py
├── automation.py
├── dataset.csv
├── requirements.txt
├── accuracy_graph.png
├── loss_graph.png
├── dog.png
├── cat.png
├── output_dog.png
├── output_cat.png
├── email_classifier_model.h5
├── vectorizer.pkl
└── README.md
```

---

## Future Improvements

* Flask-based Web Interface
* Higher Accuracy NLP Models
* LSTM / BERT Integration
* Live Camera Object Detection
* Real-time AI Dashboard

---

## Author

Made with ❤️ by Priya Pandey.

Interested in AI, Machine Learning, NLP, and Full-Stack Development.

Let’s connect on [LinkedIn](https://www.linkedin.com/in/priya-pandey-4b513b288/)
