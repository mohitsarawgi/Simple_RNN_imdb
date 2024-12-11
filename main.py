import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Input, SimpleRNN
from tensorflow.keras.utils import pad_sequences # Import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
import streamlit as st

# Load the IMDB dataset
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}


# Load the pretrainded model
model = load_model('simple_rnn_model.h5')


def decode_review(encoded_review):
  return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])


# Function to preprocess user input
def preprocess_text(text):
  words =text.lower().split()
  encoded_review = [word_index.get(word, 2) + 3 for word in words]
  padded_review =  sequence.pad_sequences([encoded_review], maxlen=500)
  return padded_review


  # Creating prediction function

def predict_sent(review):
  preprocesss_input = preprocess_text(review)
  prediction = model.predict(preprocesss_input)
  sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'
  return sentiment, prediction[0][0]


## Streamlit app
st.title('ImDB Movie Review sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')


# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
  preprocessed_input = preprocess_text(user_input)

# Make Prediction
  prediction = model.predict(preprocessed_input)
  sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'

  # Display the result
  st.write(f'Sentiment: {sentiment}')
  st.write(f'Prediction Score: {prediction[0][0]}')
else:
  st.write('Plese enter a movie review')
