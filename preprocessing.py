import numpy as np
#NLP
import nltk
from nltk.tokenize import sent_tokenize
# from nltk.tag import pos_tag
from nltk.stem.snowball import SnowballStemmer #You can call SnowballStemmer, but I believe everything we are working with is in English
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.special import softmax
from transformers import AutoConfig
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

#Downloading necessary NLTK libraries if not available
nltk.download('punkt')
nltk.download('vader_lexicon')

#spacy setup
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

#Transformer setup
MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
# tokenizer = AutoTokenizer.from_pretrained(MODEL, add_prefix_space=True)
config = AutoConfig.from_pretrained(MODEL)

#NLTK Preprocessing
class NLTKPreprocessing(BaseEstimator, TransformerMixin):
    #function to fit our data
    def fit(self, X, y=None):
        return self
    
    #Transforming function that returns our preprocessed text
    def transform(self, X, y=None):
        #Tokenizing each row in a Dataframe
        tokenized_text = []
        #7/17/24: Removing this for the streamlit app
        #Initially this was for processing a large amount of reviews, but breaks when there's only one.
        #for doc in X:
        #    tokenized_text.append(sent_tokenize(doc))
        tokenized_text.append(sent_tokenize(X)) 
            
        #Stemming the now tokenized lists of text
        stemmer = SnowballStemmer(language='english') 
        stemmed_tokens = []
        for review in tokenized_text:
            sentence = [] #List variable to hold each review
            for token in review:
                #Stemming each word in the review
                sentence.append(stemmer.stem(token))
            #Appending each stemmed review 
            stemmed_tokens.append(sentence)
            
        return stemmed_tokens
    
#Building out our sentiment scoring function:
def review_sentiment(review):
    #Init Sentiment Analysis 
    sid = SentimentIntensityAnalyzer() 
    # ~~7/17/2024 BHG: Potentially remove this for Streamlit app. The following lines were needed in the notebook analysis, but not production app
    # counter variables for getting the average score
    # count = 0.0
    score = 0.0
    #Getting the sentiment of each sentence in the review
    for sentence in review:
        sentence_polarity = sid.polarity_scores(sentence)
        score += sentence_polarity['compound']
    #     # count += 1

    #calculating the overall sentiment by way of the mean of each polarity score
    review_score = round(score, 4)
    # review_score = round((score / count), 4)
    return sentence_polarity

#Creating our Scikit-Learn Transformer for getting each review's sentiment:
class VaderSentimentScorer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y = None):
        return self
    #Transformer function that calculates each review's sentiment
    def transform(self, X, y = None):
        sentiment_scores = []
        for doc in X:
            #Getting the sentiment score for the review
            sentiment = review_sentiment(doc)
            sentiment_scores.append(sentiment)
        
        return sentiment_scores
    
#function to group scores into positive, neutral, or negative
def score_classifier(sentiment_score, pos_threshold = 0.3, neg_threshold = -0.3): #Setting boundary thresholds to give flexibility when calling the function
    if sentiment_score > pos_threshold:
        return 'Positive'
    elif sentiment_score < neg_threshold:
        return 'Negative'
    else:
        return 'Neutral'
    
#spaCy-based sentiment scoring function
#Potentially build this as a class
def spacy_sentiment(review):
    pass
    
    
#Transformer-based Scoring pipeline
class BertScorer(BaseEstimator, TransformerMixin): 
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, model, tokenizer, y = None):
        #Encoding the pre-tokenized tokens
        scores = []
        labels = []
        for i in X:
            encoded_input = tokenizer(i, return_tensors='pt', is_split_into_words=True, max_length=512, truncation=True)
            output = model(**encoded_input)
            score = output[0][0].detach().numpy()
            score = softmax(score)
            #Getting the rank of this score
            ranking = np.argsort(score)
            ranking = ranking[::-1]
            label = config.id2label[ranking[0]]
            scores.append(score)
            labels.append(label)
        
        return labels
    