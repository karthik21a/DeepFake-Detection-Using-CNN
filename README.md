# DeepFake-Detection-Using-CNN

## 📌 Project Overview

This project detects **DeepFake images** using a **Convolutional Neural Network (CNN)**, helping users identify whether an uploaded image is **real or fake**. It uses **TensorFlow, Keras, Flask**, and a simple **web interface** for user-friendly DeepFake detection.

---

## 🚀 Features

✅ DeepFake detection using a trained CNN model  
✅ Flask web interface for image upload and prediction  
✅ Supports real-time single-image detection  
✅ Saves model as `.h5` for easy reloading and inference  
✅ IEEE-style documentation for academic reference

---

## 🛠️ Tech Stack

- **Python 3.8+ / 3.10+**
- **TensorFlow, Keras** (deep learning)
- **Flask** (web framework)
- **OpenCV, NumPy** (image processing)
- **Matplotlib** (visualization, optional)
- **Anaconda (Recommended for environment management)**

---

## ⚙️ Setup Instructions – DeepFake-Detection-Using-CNN

Follow these steps precisely to set up and run your DeepFake Detection project locally.

---

### 1️⃣ Clone the Repository

Open VS Code terminal / Anaconda prompt / Command Prompt:

```bash
git clone https://github.com/your-username/DeepFake-Detection-Using-CNN.git
cd DeepFake-Detection-Using-CNN


conda create -n deepfake-env python=3.10
conda activate deepfake-env
python -m venv venv
# Activate:
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install --upgrade pip
pip install tensorflow keras flask numpy opencv-python matplotlib
```

# run applications

python train_cnn.py
python app.py

# problems faced

TensorFlow errors: Use pip install tensorflow==2.13
Missing OpenCV: pip install opencv-python
Missing Flask: pip install flask
Deactivate environment after use:
conda deactivate (Anaconda)
deactivate (venv)



