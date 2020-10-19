import streamlit as st
import pandas as pd
import numpy as np
import joblib
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

st.title("Hate Speech Detection Dashboard")

st.header('About')
st.markdown('*	A prototype type for a hate speech detection dashboard.\n*	Built on a machine learning model designed to detect hate speech in Twitter data.')


# Set page title

# Single tweet classification
# Set subheader
st.header('Single tweet classification')
st.markdown('Try out the classification algorithm:')

# Get input from the user
hs_text = st.text_area('Enter text here and click below to classify.', value='#westandtogether')

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
          
# Data set classification.

st.header('Data Set Classification')
st.markdown('*	Classifies a data set in the form of a csv file.\n* Three example files are currently available for testing: dataset1.csv, dataset2.csv and dataset3.csv\n*	Future versions of the application will include the ability to upload dataset files.\n*	Warning: the datasets contain content that some users may find ofennsive or disturbing.')

filename = st.text_input('Enter a filename:', 'dataset1.csv')

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

		# Hate speech word cloud

		stopwords = set(STOPWORDS)
		hate_speech = tweet_data[tweet_data.HS == 1]['text']

		st.subheader("Hate Speech Cloud")
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
		st.pyplot(fig)

		# Table of classifed tweets

		st.subheader("Classification Table")
		tweet_data['Hate Speech?'] = np.where(tweet_data['HS']==0, 'No', 'Yes')
		classification_table = tweet_data[['text','Hate Speech?']]
		classification_table.columns = ['Tweet', 'Hate Speech?']
		st.table(classification_table) # All tweets
		#st.table(classification_table[:20]) # All tweets - top 20







