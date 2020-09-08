import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import os
import pandas as pd
#import re
#import string
#nltk.download('stopwords')
# from PIL import Image


# Set page title

st.title("Hate Speech Detection App")

# Single tweet classification
# Set subheader
# https://blog.jcharistech.com/2019/11/14/building-a-news-classifier-machine-learning-app-with-streamlit/
st.subheader('Single tweet classification')

# Get input from the user
hs_text = st.text_area("Enter Text")

# Load vectorizer

#with st.spinner('Loading vectorizer...'):
hs_vectorizer = open("vect.pkl", "rb")
hs_tfidf=joblib.load(hs_vectorizer)

# Load classification model
# "rb" mode opens the file in binary format for reading
#with st.spinner('Loading classification model...'):
model= open("hsmod.pkl", "rb")
hsmod_clf=joblib.load(model)

# Add prediction labels

#prediction_labels = {0:'Not Hate Speech', 1:'Hate Speech'}

# Predict and display results

if st.button("Classify Tweet"):
	#st.text("Original text :\n{}".format(hs_text))
	vect_text = hs_tfidf.transform([hs_text]).toarray()
	prediction = hsmod_clf.predict(vect_text)
	probability = hsmod_clf.predict_proba(vect_text)
	#st.write(prediction[0])
	#final_result = prediction_labels[prediction[0]]
	if prediction[0] == 0:
		st.success("Not Hate Speech ({} % probability)".format(round(probability[0][0]*100),0))
	if prediction[0] == 1:
		st.error(" Hate Speech ({} % probability)".format(round(probability[0][1]*100),0))

# https://discuss.streamlit.io/t/upload-files-to-streamlit-app/80

st.subheader('Classify a data set')
filename = st.text_input('Enter a csv file path:')

#filename = 'hateval2019_en_dev.csv'
@st.cache
def get_data():
    return pd.read_csv(filename)

if st.button("Classify Data Set"):
	df = get_data()
	tweets = df['text'][0]
	st.write(tweets)

#if filename != '':
#	with open(filename) as input:
#		st.text(input.read())
#all_data = pd.read_csv(filename)


