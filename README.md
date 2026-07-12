<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=30&duration=3500&pause=1000&color=6C63FF&center=true&vCenter=true&width=900&lines=MultiModal+AI+Detection+Platform;Email+Spam+Detection+using+TensorFlow;YOLOv8+Object+Detection;Flask+%7C+NLP+%7C+Computer+Vision" />
</p>

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,flask,tensorflow,opencv,html,css,js,git,github,vscode" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-98.48%25-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Framework-Flask-black?style=for-the-badge&logo=flask"/>
  <img src="https://img.shields.io/badge/YOLOv8-Object%20Detection-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/TensorFlow-NLP-orange?style=for-the-badge&logo=tensorflow"/>
</p>

---

# MultiModal AI Detection Platform

A Flask-based AI web application that combines **Natural Language Processing (NLP)** and **Computer Vision** into a unified platform. It enables users to perform **Email Spam Detection** and **YOLOv8-based Object Detection** through a clean, responsive, and interactive dashboard with support for **batch analysis**.

---

## <img src="https://cdn-icons-png.flaticon.com/512/1828/1828919.png" width="22"> Features

### Email Spam Detection

- Detect spam and non-spam emails using a TensorFlow-based NLP model.
- Analyze **up to 10 emails** in a single request.
- Advanced preprocessing including:
  - TF-IDF Vectorization
  - Text Cleaning
  - Lowercase Conversion
  - Stopword Removal
  - Stemming
- Input validation for invalid email text.

### Image Object Detection

- Detect multiple objects using **YOLOv8**.
- Analyze **up to 10 images** simultaneously.
- Generate annotated detection images automatically.
- Display **per-image object summaries** (e.g., Person ×3, Dog ×1).
- Supports multiple object detection within the same image.

### User Interface

- Modern Flask-based dashboard.
- Dedicated pages for:
  - Dashboard
  - Email Detection
  - Image Detection
  - About
- Light/Dark Mode with theme persistence.
- Responsive and user-friendly design.

---

## <img src="https://cdn-icons-png.flaticon.com/512/2103/2103633.png" width="22"> Technologies Used

| Category | Technologies |
|----------|--------------|
| Programming | Python |
| Backend | Flask |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | YOLOv8, OpenCV |
| NLP | TF-IDF, NLTK |
| Machine Learning | Scikit-learn |
| Frontend | HTML, CSS, JavaScript |

---

## <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="22"> Model Performance

- Trained on a **real-world dataset of approximately 5.9K emails**.
- Removed duplicate records and balanced the dataset before training.
- Implemented Train-Test Split for evaluation.
- Achieved **98.48% Test Accuracy**.
- Generated:
  - Accuracy Graph
  - Loss Graph
  - Confusion Matrix
- Saved trained model and TF-IDF vectorizer for efficient inference.

---

## <img src="https://cdn-icons-png.flaticon.com/512/942/942748.png" width="22"> Application Preview

> Replace these placeholders with screenshots of your application.

| Dashboard |
|-----------|
| ![](screenshots/dashboard.png) |

| Email Detection |
|-----------------|
| ![](screenshots/email.png) |

| Image Detection |
|-----------------|
| ![](screenshots/image.png) |

| Detection Result |
|------------------|
| ![](screenshots/result.png) |

---

## <img src="https://cdn-icons-png.flaticon.com/512/5956/5956592.png" width="22"> Project Structure

```text
MultiModal-AI-Detection-Platform/
│
├── app.py
├── train_model.py
├── text_classifier.py
├── yolo_detector.py
├── requirements.txt
├── dataset.csv
│
├── models/
│   ├── email_classifier_model.h5
│   └── vectorizer.pkl
│
├── static/
│   ├── style.css
│   ├── script.js
│   └── ...
│
├── templates/
│   ├── dashboard.html
│   ├── email.html
│   ├── image.html
│   └── about.html
│
└── README.md
```

---

## <img src="https://cdn-icons-png.flaticon.com/512/724/724933.png" width="22"> Installation

Clone the repository

```bash
git clone https://github.com/ppandey1017/MultiModal-AI-Detection-Platform.git
```

Move into the project directory

```bash
cd MultiModal-AI-Detection-Platform
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## <img src="https://cdn-icons-png.flaticon.com/512/1006/1006771.png" width="22"> Train the Email Model

```bash
python train_model.py
```

This will:

- Train the spam detection model.
- Save the trained model.
- Save the TF-IDF vectorizer.
- Generate accuracy and loss graphs.

---

## <img src="https://cdn-icons-png.flaticon.com/512/1048/1048953.png" width="22"> Run the Application

```bash
python app.py
```

Open your browser and visit

```text
http://127.0.0.1:5000
```

---

## <img src="https://cdn-icons-png.flaticon.com/512/2721/2721268.png" width="22"> Workflow

### Email Detection

1. Open the Email Detection page.
2. Enter one or more email messages.
3. Click **Analyze All**.
4. View spam/non-spam predictions.

### Image Detection

1. Upload one or more images.
2. Click **Analyze All**.
3. View annotated output images.
4. Check the object summary generated for each uploaded image.

---

## <img src="https://cdn-icons-png.flaticon.com/512/5956/5956595.png" width="22"> Future Enhancements

- OCR (Image to Text)
- Speech-to-Text
- Image Caption Generation
- Face & Emotion Detection
- Plant Disease Detection
- Cloud Deployment
- Additional AI modules

---

## <img src="https://cdn-icons-png.flaticon.com/512/3135/3135768.png" width="22"> Author

**Priya Pandey**

B.Tech (Electronics & Communication Engineering with AI)  
Indira Gandhi Delhi Technical University for Women (IGDTUW)

<p>
<a href="https://github.com/ppandey1017">
<img src="https://skillicons.dev/icons?i=github" height="45"/>
</a>

<a href="https://www.linkedin.com/in/priya-pandey-4b513b288/">
<img src="https://skillicons.dev/icons?i=linkedin" height="45"/>
</a>
</p>

---

<p align="center">
<img src="https://readme-typing-svg.demolab.com?font=Poppins&size=22&duration=3500&pause=1000&color=6C63FF&center=true&vCenter=true&width=650&lines=Thank+you+for+visiting+my+project!;If+you+liked+it%2C+please+consider+giving+it+a+Star!" />
</p>
