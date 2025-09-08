# FreshGuard – AI-Powered Produce Freshness Detection

![FreshGuard Logo](https://img.shields.io/badge/FreshGuard-AI%20Powered-green?style=for-the-badge&logo=tensorflow)

**FreshGuard** is an intelligent and user-friendly application designed to automatically detect the freshness of fruits and vegetables using computer vision and deep learning techniques. The system classifies produce into three categories — **fresh**, **slightly spoiled**, and **spoiled** — based on image inputs provided by users.

## 🎯 Project Overview

The primary goal is to help reduce food waste, ensure healthier consumption, and streamline inventory management for households, grocery stores, and food delivery platforms. The project leverages **Convolutional Neural Networks (CNNs)** implemented with **TensorFlow/Keras**, and features an interactive interface built using **Streamlit**.

## ✨ Key Features

- **🔍 Image Classification**: Accurately categorizes produce into freshness levels using deep learning
- **⚡ Real-Time Prediction**: Provides instant results through an intuitive web interface
- **📊 Model Evaluation**: Uses metrics such as Accuracy, Precision, Recall, F1-score, and Confusion Matrix
- **👤 User Experience**: Simplified interface that requires no technical expertise to operate
- **📈 Scalable Design**: Built to accommodate larger datasets and extend to additional produce types

## 🛠️ Technologies Used

### Backend & AI
- **Python** - Core programming language
- **TensorFlow & Keras** - Deep learning model training and inference
- **OpenCV & Pillow** - Image preprocessing and manipulation
- **Scikit-learn** - Evaluation metrics and analysis

### Frontend & Interface
- **Streamlit** - Interactive web application framework

### Additional Tools
- **NumPy & Pandas** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip package manager
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/freshguard.git
   cd freshguard
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Usage

1. Open your browser and navigate to the provided local URL (typically `http://localhost:8501`)
2. Upload an image of a fruit or vegetable using the file uploader
3. Wait for the model to process the image
4. View the freshness classification result and confidence score
5. Make informed decisions based on the analysis

## 📁 Project Structure

```
FreshGuard/
├── __pycache__/          # Python cache files
├── .git/                 # Git version control
├── data/                 # Dataset directory
├── models/               # Trained model files
├── src/                  # Source code directory
├── Test Images - FreshGuard/  # Test image samples
├── .gitattributes        # Git attributes configuration
├── .gitignore           # Git ignore file
├── app.py               # Main Streamlit application
├── freshGuard.py        # Core FreshGuard logic
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
└── runtime.txt          # Python runtime specification
```

## 🎯 How It Works

1. **Image Upload**: Users upload an image of produce through the web interface
2. **Preprocessing**: The image is resized, normalized, and prepared for model input
3. **Prediction**: The trained CNN model analyzes the image features
4. **Classification**: The model outputs one of three categories:
   - 🟢 **Fresh**: Safe for consumption
   - 🟡 **Slightly Spoiled**: Use with caution or consume soon
   - 🔴 **Spoiled**: Not recommended for consumption
5. **Results Display**: The interface shows the prediction with confidence percentage

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 92.5% |
| Precision | 91.8% |
| Recall | 90.3% |
| F1-Score | 91.0% |

## 🌍 Applications

- **🏪 Grocery Stores**: Reduce produce waste and maintain quality standards
- **🏠 Households**: Make informed consumption decisions and reduce food waste
- **🚚 Food Delivery**: Ensure fresh ingredients reach customers
- **🎓 Education**: Demonstrate practical applications of AI and computer vision
- **📦 Inventory Management**: Automate quality control processes

## 🔬 Technical Challenges Solved

- **Multi-stage Spoilage Detection**: Distinguishing between slightly spoiled and fully spoiled produce
- **Image Variability**: Handling different lighting conditions, angles, and textures
- **Real-time Performance**: Optimizing model inference for quick predictions
- **User Accessibility**: Creating an intuitive interface for non-technical users

## 🚀 Future Enhancements

- [ ] **Mobile App Development**: Native iOS and Android applications
- [ ] **Video Analysis**: Real-time spoilage detection through video streams
- [ ] **Additional Produce Types**: Expand to dairy products and other perishables
- [ ] **IoT Integration**: Connect with smart storage systems
- [ ] **Alert Systems**: Automated notifications for inventory management
- [ ] **Ensemble Methods**: Combine multiple models for improved accuracy

## 🤝 Contributing

We welcome contributions to FreshGuard! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 coding standards
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📋 Requirements

```
streamlit==1.28.0
tensorflow==2.13.0
opencv-python==4.8.0
pillow==10.0.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the excellent deep learning framework
- Streamlit team for the intuitive web app framework
- Contributors to the open-source computer vision community
- Dataset providers for produce image collections

## 💡 Impact Statement

FreshGuard contributes to:
- **🌱 Environmental Sustainability**: Reducing food waste through early spoilage detection
- **💪 Health Promotion**: Helping users make informed consumption decisions
- **📈 Economic Efficiency**: Minimizing losses in food supply chains
- **🔬 AI Accessibility**: Making advanced technology available to everyday users

---

**Made with ❤️ for a more sustainable future**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=flat-square&logo=streamlit)](https://streamlit.io)