import pickle

from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the tokenizer and model
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)
model = load_model('model.h5')


# Class to represent feedback
class Feedback:
    def __init__(self, id, userName, feedbackText):
        self.id = id
        self.userName = userName
        self.feedbackText = feedbackText


# Previous feedback data (you can replace this with a database query)
previous_feedback = [
    Feedback(1, 'John Doe', 'Great app!'),
    Feedback(2, 'Jane Smith', 'Very helpful.'),
    # Add more feedback objects as needed
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    sequences = loaded_tokenizer.texts_to_sequences([text])
    max_sequence_length = 100
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    prediction = model.predict(padded_sequences)[0][0]

    is_cyberbullying = prediction >= 0.5
    result = "Cyberbullying Detected!" if is_cyberbullying else "No Cyberbullying Detected"
    result_color = "red" if is_cyberbullying else "green"

    return jsonify({'text': result, 'color': result_color})


@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    user_name = request.form.get('userName', '')
    feedback_text = request.form.get('feedbackText', '')

    # Append the user name and feedback text to a text file
    with open('feedback.txt', 'a') as feedback_file:
        feedback_file.write(f'{user_name}: {feedback_text}\n')

    return jsonify({'status': 'success'})


@app.route('/get-feedback', methods=['GET'])
def get_feedback():
    feedback_list = [{'id': feedback.id, 'userName': feedback.userName, 'feedbackText': feedback.feedbackText}
                     for feedback in previous_feedback]
    return jsonify(feedback_list)


# Route to handle feedback voting
@app.route('/vote-feedback', methods=['POST'])
def vote_feedback():
    # Implement the logic to handle feedback voting
    # Retrieve feedback ID and vote type from the request data
    feedback_id = int(request.form.get('feedbackId', 0))
    vote_type = request.form.get('voteType', '')

    # You can update the feedback data or perform any other action based on the vote
    print(f'Voted {vote_type} for feedback with ID {feedback_id}')

    return jsonify({'status': 'success'})


if __name__ == '__main__':
    app.run(debug=True)
