# Sentiment Analysis Project

A Python project that performs sentiment analysis on text data using **Scikit-learn** and provides a simple **Tkinter GUI** for user interaction.  
This project trains a sentiment classification model and allows users to test it via a user-friendly interface.

---

## 📂 Project Structure

```

SentimentAnalysisProject/
│── data/                  # Dataset folder (CSV/text files for training)
│── model/                 # Saved trained model + vectorizer
│── requirements.txt       # Required Python packages
│── README.md              # Project documentation
│── train\_model.py         # Script to train the sentiment model
│── app.py                 # Tkinter GUI to test the model

````

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/SentimentAnalysisProject.git
   cd SentimentAnalysisProject
````

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

1. Train the model:

   ```bash
   python train_model.py
   ```

   This will create and save the model + vectorizer inside the `model/` folder.

2. Run the GUI application:

   ```bash
   python app.py
   ```

   Enter a sentence in the input box, and the model will classify it as **Positive**, **Negative**, or **Neutral**.

---

## 📦 Requirements

* Python 3.x
* pandas
* scikit-learn
* tkinter (usually comes with Python)


---

## 📜 License

This project is open-source and free to use for learning purposes.

```
