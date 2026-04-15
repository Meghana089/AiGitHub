import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib

# Load the training dataset
def load_data(filepath):
    """Load dataset from text file with format: text;emotion"""
    texts = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and ';' in line:
                # Split by the last semicolon to handle text with semicolons
                parts = line.rsplit(';', 1)
                text = parts[0]
                emotion = parts[1]
                texts.append(text)
                labels.append(emotion)
    
    return texts, labels

# Create a DataFrame
def create_dataframe(texts, labels):
    """Create a pandas DataFrame from texts and labels"""
    df = pd.DataFrame({
        'text': texts,
        'emotion': labels
    })
    return df

# Analyze dataset
def analyze_dataset(df):
    """Print dataset statistics"""
    print("Dataset Statistics")
    print("=" * 50)
    print(f"Total samples: {len(df)}")
    print(f"\nEmotion distribution:")
    print(df['emotion'].value_counts())
    print(f"\nPercentage distribution:")
    print(df['emotion'].value_counts(normalize=True) * 100)
    print(f"\nAverage text length: {df['text'].str.len().mean():.2f} characters")
    print()

# Train model
def train_model(texts, labels, test_size=0.2, random_state=42):
    """Train a sentiment/emotion classification model"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Create pipeline with TF-IDF and Naive Bayes
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ])
    
    # Train model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Training Complete!")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, X_test, y_test

# Predict emotion
def predict_emotion(model, text):
    """Predict emotion for a given text"""
    prediction = model.predict([text])[0]
    confidence = max(model.predict_proba([text])[0])
    return prediction, confidence

# Main execution
if __name__ == "__main__":
    # Load data
    dataset_path = "train (1).txt"
    
    if os.path.exists(dataset_path):
        print("Loading dataset...")
        texts, labels = load_data(dataset_path)
        
        # Create DataFrame
        df = create_dataframe(texts, labels)
        
        # Analyze dataset
        analyze_dataset(df)
        
        # Train model
        model, X_test, y_test = train_model(texts, labels)
        
        # Save the trained model
        model_path = "emotion_classifier_model.pkl"
        print(f"\nSaving model to {model_path}...")
        joblib.dump(model, model_path)
        print(f"Model saved successfully!")
        
        # Example predictions
        print("\n" + "=" * 50)
        print("Example Predictions:")
        print("=" * 50)
        
        test_sentences = [
            "I feel very happy and excited",
            "I am so sad and depressed",
            "I am feeling angry and frustrated",
            "I am scared and afraid"
        ]
        
        for sentence in test_sentences:
            emotion, confidence = predict_emotion(model, sentence)
            print(f"Text: {sentence}")
            print(f"Emotion: {emotion} (Confidence: {confidence:.4f})\n")
    else:
        print(f"Error: Dataset file '{dataset_path}' not found!")
