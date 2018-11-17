#import libraries
import re
import pickle
import nltk
from nltk.corpus import stopwords 
from sklearn.datasets import load_files
nltk.download('stopwords')

#import Dataset
reviews = load_files('txt_sentoken')
X,y=reviews.data,reviews.target

with open('X.pickle','wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle','wb') as f:
    pickle.dump(y,f)
    
#unpickling the dataset
with open('X.pickle','rb') as f:
    pickle.load(f)
    
with open('y.pickle','rb') as f:
    pickle.load(f)
    
#creating the corpus
corpus=[]
for i in range(0,len(X)):
    reviews= re.sub(r'\W',' ',str(X[i]))
    reviews=reviews.lower()
    reviews=re.sub(r'\s+[a-z]\s+', ' ',reviews)
    reviews=re.sub(r'^[a-z]\s+', ' ',reviews)
    reviews=re.sub(r'\s+', ' ',reviews)
    corpus.append(reviews)
    
#BagOfWords 
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(max_features=2000,min_df=3,max_df=0.6)
X=vectorizer.fit_transform(corpus).toarray()

#Tfidf
from sklearn.feature_extraction.text import TfidfTransformer                      
transformer=TfidfTransformer()
X=transformer.fit_transform(X).toarray()

#Splitting the dataset into training and test data
from sklearn.model_selection import train_test_split
text_train,text_test,sent_train,sent_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Training the model using Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression()
classifier.fit(text_train,sent_train)

#Finding the test results
sent_pred=classifier.predict(text_test)

#Inspect the data using Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(sent_test,sent_pred)

#pickling the classifier
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
with open('bagofwords.pickle','wb') as f:
    pickle.dump(vectorizer,f)
    
with open('tfidf.pickle','wb') as f:
    pickle.dump(transformer,f)
    
with open('classifier.pickle','rb') as f:
   clf= pickle.load(f)

with open('bagofwords.pickle','rb') as f:
    bw=pickle.load(f)

with open('tfidf.pickle','rb') as f:
    tfidf=pickle.load(f)

#first input
sample=["Your presentation is terrific and awesome"]
sample=bw.transform(sample).toarray()
sample=tfidf.transform(sample).toarray()
if clf.predict(sample)==1:
    print("Positive Response")
else:
    print("Negative Response")
    
#second input
sample=["The Weather is terrible"]
sample=bw.transform(sample).toarray()
sample=tfidf.transform(sample).toarray()
if clf.predict(sample)==1:
    print("Positive Response")
else:
    print("Negative Response")
    
    