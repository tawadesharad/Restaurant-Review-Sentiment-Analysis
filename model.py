import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


review_df = pd.read_csv('https://raw.githubusercontent.com/tawadesharad/Zomato-Restaurant-Clustering-And-Sentiment-Analysis-capstone-project-4/main/Zomato%20Restaurant%20reviews.csv')

review_df.loc[review_df['Rating'] == 'Like'] = np.nan
review_df['Rating']= review_df['Rating'].astype('float64')
review_df['Rating'].fillna(3.6, inplace=True)

# split metadata column into 2 columns i.e. Reviews and followers
review_df['Reviews'],review_df['Followers']= review_df['Metadata'].str.split(',').str
review_df['Reviews'] = pd.to_numeric(review_df['Reviews'].str.split(' ').str[0])

# Converting Time column into Time, Year, Month, Hour
review_df['Time']=pd.to_datetime(review_df['Time'])
review_df['Year'] = pd.DatetimeIndex(review_df['Time']).year
review_df['Month'] = pd.DatetimeIndex(review_df['Time']).month
review_df['Hour'] = pd.DatetimeIndex(review_df['Time']).hour
review_df = review_df.drop(['Metadata'], axis =1)

# Replacing followers and reviews null values into 0
review_df['Followers'].fillna(0,inplace=True)
review_df['Reviews'].fillna(0,inplace=True)

# we can drop the remaining missing data
review_df.dropna(inplace=True)

review_df.reset_index(inplace = True)
review= review_df.Review

def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)

review_df['Review'] = review_df['Review'].apply(remove_punctuation)

# Removing stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# extracting the stopwords from nltk library
sw = stopwords.words('english')

def stopwords(text):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)

review_df['Review'] = review_df['Review'].apply(stopwords)

# Lemmatization
review=review_df.Review
import spacy
nlp = spacy.load('en_core_web_sm')
def lemmatization_(text):
  for index,x in enumerate(text):
    doc = nlp(x)  
    l=list()
    for word in doc:
        l.append(word.lemma_)
    text[index]=' '.join(l)
  return text

review=lemmatization_(review)

# remove_all_extra_spaces
def remove_spaces (text):
  '''removes all extra space from the text
  '''
  for index,x in enumerate(text):
    text[index]=" ".join(x.split())
  return text

review=remove_spaces(review)

# Remove non letters
import re
regex = re.compile('[^a-zA-Z]')
def remove_non_leters(text):
  '''used to remove all non leters form the list 
  '''
  text=[regex.sub(' ', x) for x in text]
  return text

review=remove_non_leters(review)

review_df['Review']=review

#function to removing words greater than 45 and less than 2
def len_less_than2(review):
  review=" ".join([i for i in review.split() if len(i)>2])
  review=" ".join([i for i in review.split() if len(i)<=45])
  return review

#removing words greater than 45 and less than 2
review_df['Review']=review_df['Review'].apply(lambda x:len_less_than2(x))


from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#Create a function to get the subjectivity
def subjectivity(text): 
    return TextBlob(text).sentiment.subjectivity

#Create a function to get the polarity
def polarity(text): 
    return TextBlob(text).sentiment.polarity

#Create two new columns
review_df['Subjectivity'] = review_df['Review'].apply(subjectivity)
review_df['Polarity'] = review_df['Review'].apply(polarity)

#Create a function to compute the negative, neutral and positive analysis
def getAnalysis(score):
    if score <=0:
        return 0
    else:
        return 1

review_df['Analysis'] = review_df['Polarity'].apply(getAnalysis)   

import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

corpus = []
for i in range(len(review_df)):
    review = re.sub('[^a-zA-Z]', ' ', review_df['Review'][i])
    review = review.lower()
    review = review.split()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not') 
    #remove negative word 'not' as it is closest word to help determine whether the review is good or not 
    review = [stemmer.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = review_df['Analysis']

import joblib
# Creating a pickle file for the CountVectorizer model
joblib.dump(cv, "cv.pkl")

from sklearn.model_selection import train_test_split

# Model Building
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.linear_model import LogisticRegression

# Fitting Naive Bayes to the Training set
# classifier = MultinomialNB(alpha=0.2)
classifier= LogisticRegression(fit_intercept=True, max_iter=10000)
classifier.fit(X_train, y_train)

# Creating a pickle file for the Multinomial Naive Bayes model
joblib.dump(classifier, "model.pkl")