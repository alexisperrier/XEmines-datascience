import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

df = pd.read_csv('bbc.csv')
df.repertoire.value_counts()

'''
Encoder les categories de la variable cible
'''
le = LabelEncoder()
df['repertoire'] = le.fit_transform(df.repertoire)
df.repertoire.value_counts()

y = df.repertoire

'''
Vectoriser le texte avec TfidfVectorizer - TF-IDF
'''

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words ='english')
X = vectorizer.fit_transform(df.contenu)

'''
Entrainer un modele Naive Bayes Multinomial
'''

clf = MultinomialNB(alpha = 0.1)
clf.fit(X,y)
yhat = clf.predict(X)

'''
La classification est presque parfaite
'''
clf.score(X,y)
confusion_matrix(yhat, y)
