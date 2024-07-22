import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st 
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
#preprocessing containing the sentiment scorers
import preprocessing
#For pipeline building
from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 


#Transformer setup
MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
tokenizer = AutoTokenizer.from_pretrained(MODEL, add_prefix_space=True)
config = AutoConfig.from_pretrained(MODEL)
#Setting this up with Torch
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def main():
    #initializing the preprocessing objects
    nltk_preprocessor = preprocessing.NLTKPreprocessing()
    vader_scorer = preprocessing.VaderSentimentScorer()
    bert_scorer = preprocessing.BertScorer()

    #Setting up the streamlit app
    st.title('Sentiment Analysis Comparison')
    st.subheader('NLTK vs. Roberta')

    text = st.text_input('Text for Sentiment Analyzer')
    clicked = st.button('Submit')

    #Setting up each column (for NLTK vs Roberta)
    col1, col2, col3 = st.columns(3, gap='large')
    col1.header('Vader')
    col2.header('spaCy')
    col3.header('Bert')
        
    #tokenizing and scoring the text upon submission
    if clicked == True:
        logging.basicConfig(filename='streamlit.log', level=logging.DEBUG,
                    format='%(asctime)s - %(message)s', datefmt='%d-%b')
        
        tokens = nltk_preprocessor.fit_transform(text)

        with col1:
            with st.spinner('Scoring the Sentiment of the Text...'):
                vader_score = vader_scorer.fit_transform(tokens)
                #converted_vader = preprocessing.score_classifier(vader_score)
                st.write(vader_score)
                
                
        with col2:
            with st.spinner('Scoring the Sentiment of the Text...'):
                spacy_score = preprocessing.spacy_sentiment(text)
                st.write(spacy_score)
        
        with col3:
            with st.spinner('Scoring the Sentiment of the Text...'):
                bert_score = bert_scorer.transform(tokens, model=model, tokenizer=tokenizer)
                #converted_bert = preprocessing.score_classifier(bert_score)
                st.write(bert_score)
        
        #need to work on compiling the scores into something comparable
        list_of_scores = [vader_score[0]['compound'], spacy_score, bert_score[0]]        
        fig, ax = plt.subplots()
        ax.hist(list_of_scores)
        st.pyplot(fig)
                
if __name__ == "__main__":
    main()