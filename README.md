# ♻️ EcoSort AI: Intelligent Waste Classification

[![GitHub last commit](https://img.shields.io/github/last-commit/akshbala-01/EcoSort-AI/main)](https://github.com/akshbala-01/EcoSort-AI/commits/main)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This project implements an intelligent waste classification system using Deep Learning (Convolutional Neural Networks) and presents the results through an interactive web application built with Streamlit. The goal is to automatically sort common household waste materials to promote better recycling practices.

## ✨ Key Features

* **Custom CNN Model:** A multi-layered Convolutional Neural Network (CNN) is designed, trained, and optimized using **TensorFlow/Keras** for image recognition.
* **Six-Class Classification:** The model predicts one of six waste categories: **cardboard, glass, metal, paper, plastic, and trash**.
* **Streamlit Web Interface:** A user-friendly web application (`app.py`) allows real-time image upload and classification.
* **Git LFS:** The large trained model file (`waste_model.h5`) is managed using Git LFS to bypass the GitHub file size limit.

## 📁 Project Structure

| File/Folder | Description |
| :--- | :--- |
| `train_model.py` | Script for defining, training, and saving the CNN model. |
| `app.py` | The Streamlit web application that loads the model and runs predictions. |
| `requirements.txt` | Lists all necessary Python dependencies (TensorFlow, Streamlit, etc.). |
| `models/` | Stores the trained model file, `waste_model.h5`. |
| `dataset/` | Contains the raw image data used for training and validation. |

## 🚀 Setup and Running Locally

Follow these steps to get a local copy of the project running:

### 1. Clone the Repository

Clone the project and navigate into the directory. Git LFS should automatically download the model file.

```bash
git clone [https://github.com/akshbala-01/EcoSort-AI.git](https://github.com/akshbala-01/EcoSort-AI.git)
cd EcoSort-AI
2. Create and Activate Environment
Create a virtual environment to manage dependencies cleanly.

Bash

# Create the environment
python -m venv venv

# Activate the environment on Windows Command Prompt/PowerShell
.\venv\Scripts\activate
3. Install Dependencies
Install all required Python libraries:

Bash

pip install -r requirements.txt
4. Run the Web Application
With the environment active, run the Streamlit app. This command will load the pre-trained model and open the web application in your default browser.

Bash

streamlit run app.py

