# SENTINEL AI
## Deepfake Audio Detection System

SENTINEL AI detects potentially real or fake audio using a trained PyTorch CNN model.

### Features
- Real/Fake audio classification
- Streamlit web interface
- User login and signup
- Analysis history
- PDF report generation
- Model performance dashboard

### Technologies
Python, PyTorch, Streamlit, Librosa, NumPy, Scikit-learn, SQLite.

### Installation
1. Clone this repository.
2. Install dependencies using pip install -r requirements.txt.
3. Run the app using streamlit run app.py.

### Model Performance
Accuracy on the ASVspoof 2019 Logical Access development dataset: 98.30%.

Real-class recall: 87.76%.
Fake-class recall: 99.50%.

These are development-set results. Real-world performance may vary.

### Author
Harsh Patel
GitHub: https://github.com/ha6374

### Disclaimer
This project is intended for educational and research purposes. Predictions are not definitive proof that audio is real or fake.
