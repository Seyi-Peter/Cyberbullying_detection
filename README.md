Cyberbullying Detection Web App

This is a Flask-based web application that uses a trained deep learning model (CNN-GRU) to detect cyberbullying in user-submitted text. It also allows users to submit feedback, and view or vote on existing feedback.

-Cyberbullying Detection using a trained Keras model.
-User Feedback Submission
-Feedback Voting
-Real-Time Prediction** via AJAX.

Model Info
- Framework: TensorFlow / Keras
- Input: Padded text sequences
- Output: Binary prediction (Cyberbullying / Not)

Project Structure
│
├── use_model.py # Flask backend
├── model.h5 # Trained Keras model
├── best_model.h5 A 3X trained keras model
├── tokenizer.pickle # Tokenizer used for training
├── feedback.txt # Stores submitted feedback (optional)
│
├── templates/
│ └── index.html # Main HTML page
│
├── static/
│ └── [your assets here] # (CSS, JS, etc.)
│
└── requirements.txt # Python dependencies


---
Installation

1. Clone the repo
   ```bash
   git clone https://github.com/Seyi-Peter/cyberbullying-detection.git
   cd cyberbullying-detection

2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Run the app:
   python use_model.py

5. Open your browser and visit:
   http://127.0.0.1:5000/

The app will return:

✅ "No Cyberbullying Detected" (green)

❌ "Cyberbullying Detected!" (red)

---
Future Improvements
Store feedback and votes in a real database (e.g., SQLite or PostgreSQL).

-User authentication.

-Model training from the web interface.

-Better UI with Bootstrap or TailwindCSS.

---
License
This project is open-source and free to use for educational purposes.

---
👨‍💻 Author
Oluwaseyi Akinlade- Seyi-Peter
