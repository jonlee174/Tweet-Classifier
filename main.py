import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

american_tweets = pd.read_json("american.json", lines=True)
british_tweets = pd.read_json("british.json", lines=True)

american_text = american_tweets['text'].to_list()
british_text = british_tweets['text'].to_list()

all_tweets = american_text + british_text
labels = [0] * len(american_text) + [1] * len(british_text)
train_data, test_data, train_labels, test_labels = train_test_split(all_tweets, labels, test_size = 0.2, random_state = 1)

counter = CountVectorizer()
counter.fit(train_data)
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

classifier = MultinomialNB()
classifier.fit(train_counts, train_labels)
predictions = classifier.predict(test_counts)

#print(accuracy_score(test_labels, predictions))
#print(confusion_matrix(test_labels, predictions))

tweet = "I support the american Second Amendment"
tweet_counts = counter.transform([tweet])
print(classifier.predict(tweet_counts))