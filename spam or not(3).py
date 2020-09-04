import pandas as pd 
import numpy as np 
import seaborn as sns 
import sklearn 
from sklearn.model_selection import train_test_split
import nltk
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import string
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle

if __name__ == "__main__":
    

 dataset = pd.read_csv(r"F:\python new\spam or not spam\spamham 10 records.csv")
 dataset.drop_duplicates(subset = 'text' , inplace = True)
 #print(dataset.isnull().sum())

 word = {}


 def process_text(text):
     nopunc = [char for char in text if char not in string.punctuation]
     nopunc = ''.join(nopunc)
    
     clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
     return clean_words


 #dataset['vector text'] = dataset['text'].head(10).apply(process_text)
 #dataset['vector text'] = 
 #print(dataset['vector text'])




 #dataset['messages_bow'] = CountVectorizer(dataset['vector text']).fit_transform(dataset['vector text'])
 #print(dataset['messages_bow'].head())
 vectorizer = CountVectorizer(analyzer= process_text)
 vec_text = vectorizer.fit_transform(dataset['text'])

 X = vec_text
 Y = dataset['spam']
 X_train, X_test, y_train, y_test = train_test_split(X , Y, test_size = 0.20, random_state = 0)
 #print(X_train)
 from sklearn.naive_bayes import MultinomialNB
 sent_msg = {}
 classifier = MultinomialNB()
 classifier.fit(X_train, y_train)


 #pred1 = classifier.predict(X_test)
 #for i in range(0, 5):
  #       print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(y_train)[i], pred1[i]))


 with open('model.pkl', 'wb') as file:
         pickle.dump(( vectorizer, classifier), file)

 #print('Accuracy: ', accuracy_score(y_test,pred1)*100 , "%")



 #print("I")
 #print(classifier.predict(X_train))
 #Print the actual values
 #print(y_train.values)




 #print(dataset.shape)
