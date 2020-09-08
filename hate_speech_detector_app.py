import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
#import re
#import string
#nltk.download('stopwords')
# from PIL import Image


# Set page title

st.title("Hate Speech Detection App")

# Get input from the user

hs_text = st.text_area("Enter Text","Type Here")

# Load vectorizer

hs_vectorizer = open("vect.pkl", "rb")
hs_tfidf=joblib.load(hs_vectorizer)

# Load classification model
# "rb" mode opens the file in binary format for reading
# with st.spinner('Loading classification model...'):
model= open("hsmod.pkl", "rb")
hsmod_clf=joblib.load(model)

#st.info("Prediction with ML")

if st.button("Classify"):
	st.text("Original text :\n{}".format(hs_text))
	vect_text = hs_tfidf.transform([hs_text]).toarray()
	prediction = hsmod_clf.predict(vect_text)
	st.write(prediction[0])


#Preprocessing function

#def clean_text(text):
#    punct_string = string.punctuation.replace("#","") 
#    stopword = nltk.corpus.stopwords.words('english')
#    text = "".join([word for word in text if word not in punct_string])
#    text = [word.lower() for word in tokens if word not in stopword]
#    return text

# Classify individual tweet
# Set a subheader 

#st.subheader('Single tweet classification')

# Take text input from the user:





#if tweet_input != '':
    # Pre-process tweet
 #   tweet = clean_text(tweet_input)
 #   tfidf_vect=TfidfVectorizer()
  #  tfidf= vectorizer.fit(train_comments)
    #count_vect=CountVectorizer()
    #X_new_counts = count_vect.transform(sentence)
   # X_new_tfidf = tfidf_transformer.transform(X_new_counts)

	# Make predictions
 #   with st.spinner('Predicting...'):
 #   	hsmod_clf.predict(sentence)

    # Show predictions
#    label_dict = {'0': 'Not hate speech', '1': 'Hate speech'}


 #   if len(sentence.labels) > 0:
#        st.write('Prediction:')
 #       st.write(label_dict[sentence.labels[0].value] + ' with ',
#                sentence.labels[0].score*100, '% confidence')


#st.text_area("Paste text here")






# Initialising

#text_list=['text to classify']
#text_input_values=[]
#text_default_values=['I love you']
#values = []

#parameter_list=['Sepal length (cm)','Sepal Width (cm)','Petal length (cm)','Petal Width (cm)']
#parameter_input_values=[]
#parameter_default_values=['5.2','3.2','4.2','1.2']
#values=[]

#Display

#for text, text_df in zip(text_list, text_default_values):
#	values = st.text_area(label=text, value=text_df)
#	text_input_values.append(values)

#for parameter, parameter_df in zip(parameter_list, parameter_default_values):
#	values= st.sidebar.slider(label=parameter, key=parameter,value=float(parameter_df), min_value=0.0, max_value=8.0, step=0.1)
#	parameter_input_values.append(values)

#input_variables=pd.DataFrame([text_input_values],columns=text_list,)
#st.write('\n\n')

#input_variables=pd.DataFrame([parameter_input_values],columns=parameter_list,dtype=float)
#st.write('\n\n')

#if st.button("Click Here to Classify"):
#	prediction = hsmod_clf.predict(input_variables)
#	st.write(prediction)


#docs
#M_new = [user_input]
#count_vect=CountVectorizer()
#X_new_counts = count_vect.transform(docs_new)
#X_new_tfidf = tfidf_transformer.transform(X_new_counts)



#if st.button("Click Here to Classify"):
#	predicted = hsmod_clf.predict(X_new_tfidf)
#	st.write(predicted)


