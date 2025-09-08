# 🍎 FreshGuard  

FreshGuard is an AI-powered **fruit and vegetable freshness classifier** that helps detect whether produce is **Fresh**, **Slightly Spoiled**, or **Spoiled** from an image.  

Built with **TensorFlow/Keras**, **OpenCV**, and **Streamlit**, FreshGuard aims to reduce food waste and assist retailers, households, and supply chains in making better decisions about food consumption and storage.  

---

## 🚀 Features  
- ✅ Classifies fruits/vegetables into 3 categories: **Fresh**, **Slightly Spoiled**, **Spoiled**  
- ✅ User-friendly **Streamlit web interface** for quick testing  
- ✅ Pretrained deep learning model included for inference  
- ✅ Works with common produce images (fruits & vegetables)  
- ✅ Lightweight and easy to deploy  

---

## 📂 Project Structure  
```
FreshGuard/
│── app.py                # Streamlit app for UI
│── freshGuard.py          # Core classification logic
│── models/                # Saved trained model(s)
│── data/                  # Dataset (or sample images)
│── requirements.txt       # Python dependencies
│── .gitignore
│── .gitattributes
```

---

## ⚙️ Installation  

1. **Clone the repository**  
```bash
git clone https://github.com/aryanshah2109/FreshGuard.git
cd FreshGuard
```

2. **Create and activate a virtual environment**  
```bash
python -m venv venv
source venv/bin/activate     # On Linux/Mac
venv\Scripts\activate        # On Windows
```

3. **Install dependencies**  
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage  

### Run the Streamlit App  
```bash
streamlit run app.py
```

This will start a local server. Open the displayed link in your browser.  

### Example Workflow  
1. Upload an image of a fruit/vegetable.  
2. The model processes the image.  
3. Output: **Fresh / Slightly Spoiled / Spoiled** along with confidence score.  

---

## 📊 Dataset & Model  

- The model is trained on a custom dataset of fruits and vegetables at different freshness stages.  
- Uses **EfficientNet / CNN-based architecture** (update depending on your model).  
- Data preprocessing: resizing, normalization, and augmentation for robustness.  

---

## 🛠️ Tech Stack  

- 🐍 **Python 3.9+**  
- 🤖 **TensorFlow / Keras** – Deep Learning  
- 👁️ **OpenCV** – Image Processing  
- 📊 **NumPy, Pandas** – Data handling  
- 📈 **Matplotlib** – Visualization  
- 🌐 **Streamlit** – Web app  

---

## 📌 Future Improvements  
- 📷 Add real-time camera input  
- 🍏 Expand dataset to more fruit/vegetable categories  
- 📱 Mobile app integration  
- ☁️ Deploy on cloud platforms (Heroku, Streamlit Cloud, AWS)  

---

## 🤝 Contributing  
Contributions are welcome!  

1. Fork the repo  
2. Create a new branch (`feature-new`)  
3. Commit your changes  
4. Open a Pull Request  

---

## 📜 License  
This project is licensed under the **MIT License** – feel free to use and modify it.  

---

## 🙌 Acknowledgments  
- Inspired by the need to **reduce food waste**  
- Dataset prepared from [Kaggle](https://www.kaggle.com/) & custom sources  
