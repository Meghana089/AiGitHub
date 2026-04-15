from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "emotion_classifier_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
else:
    print(f"Error: Model file '{MODEL_PATH}' not found!")
    model = None

# Emotion to color mapping
EMOTION_COLORS = {
    'joy': '#FFD700',
    'sadness': '#4169E1',
    'anger': '#FF4500',
    'fear': '#8B008B',
    'love': '#FF69B4',
    'surprise': '#00CED1'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please enter some text'}), 400
        
        # Make prediction
        emotion = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        
        # Get confidence
        confidence = float(max(probabilities))
        
        # Get all emotion probabilities
        classes = model.named_steps['classifier'].classes_
        emotion_probs = {}
        for i, cls in enumerate(classes):
            emotion_probs[cls] = float(probabilities[i])
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'emotion_probabilities': emotion_probs,
            'color': EMOTION_COLORS.get(emotion, '#CCCCCC')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-samples', methods=['GET'])
def test_samples():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    samples = [
        "I feel very happy and excited today!",
        "I am so sad and depressed right now",
        "This makes me so angry and frustrated",
        "I am scared and afraid of what might happen",
        "I love you so much, you mean everything to me",
        "I am shocked and surprised by this news"
    ]
    
    results = []
    for sample in samples:
        emotion = model.predict([sample])[0]
        confidence = float(max(model.predict_proba([sample])[0]))
        results.append({
            'text': sample,
            'emotion': emotion,
            'confidence': confidence,
            'color': EMOTION_COLORS.get(emotion, '#CCCCCC')
        })
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
