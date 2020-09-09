import streamlit as st
import pandas as pd
import joblib
import os
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from twitterscraper import query_tweets
import datetime as dt

st.title("Hate Speech Detection App")

# Set page title


# Single tweet classification
# Set subheader
# https://blog.jcharistech.com/2019/11/14/building-a-news-classifier-machine-learning-app-with-streamlit/
st.subheader('Single tweet classification')

# Get input from the user
hs_text = st.text_area("Enter Text", "I love Manchester")

# Load vectorizer
# "rb" mode opens the file in binary format for reading

hs_vectorizer = open("vect.pkl", "rb")
hs_tfidf=joblib.load(hs_vectorizer)

# Load classification model

model= open("hsmod.pkl", "rb")
hsmod_clf=joblib.load(model)

# Predict and display results

if st.button("Classify Tweet"):

	vect_text = hs_tfidf.transform([hs_text]).toarray()
	prediction = hsmod_clf.predict(vect_text)
	probability = hsmod_clf.predict_proba(vect_text)

	if prediction[0] == 0:

		st.success("Not Hate Speech ({} % probability)".format(round(probability[0][0]*100),0))

	if prediction[0] == 1:

		st.error(" Hate Speech ({} % probability)".format(round(probability[0][1]*100),0))

# Twitter Query
# https://medium.com/analytics-vidhya/building-a-twitter-sentiment-analysis-app-using-streamlit-d16e9f5591f8
           
# Data set classification.

st.subheader('Data Set Classification')

filename = st.text_input('Enter a csv file path:', 'hateval2019_en_dev.csv')

#@st.cache
def get_data():
    return pd.read_csv(filename)

# Initialize empty dataframe

tweet_data = pd.DataFrame({
	'text': [],
	'HS': []
	})

if st.button("Classify Data Set"):
	with st.spinner('Detecting Hate Speech...'):
		df = get_data()
		tweets = df['text'][0:1000]

		# Add data for each tweet

		for tweet in tweets:
			if tweet in ('', ' '):
				continue
			vect_tweet = hs_tfidf.transform([tweet]).toarray()
			prediction = hsmod_clf.predict(vect_tweet)
			tweet_data = tweet_data.append({'text': tweet, 'HS': prediction[0]}, ignore_index=True)

		# Wordcloud for tweets classified as hate speech

		stopwords = set(STOPWORDS)
		hate_speech = tweet_data[tweet_data.HS == 1]['text']

		wordcloud = WordCloud(
			background_color='white',
			stopwords=stopwords,
			max_words=200,
			max_font_size=100, 
			scale=3,
			random_state=1
			).generate(str(hate_speech))
		fig = plt.figure(1, figsize=(12, 12))
		plt.axis('off')
		plt.imshow(wordcloud)
		st.pyplot()

		# Table of tweets classifed as hate speech

		st.table(hate_speech[0:50])





