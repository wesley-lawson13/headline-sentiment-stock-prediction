import re
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from data import train, test

nltk.download('stopwords')
stops = stopwords.words('english')
stops.extend([",", ".", "!", "?", "'", '"', "I", "i", "n't", "'ve", "'d", "'s"])
poswords = []
negwords = []
allwords = []

def process_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = list(set([w for w in words if w not in stops]))
    return words

for _, row in train.iterrows():
    words = process_text(row['Text'])
    if row['Label'] == 1:
        poswords.append(words)
    else:
        negwords.append(words)

    allwords.extend(words)

print("Number of positive docs:", len(poswords))
print("Number of negative docs:", len(negwords))
print("Vocabulary size:", len(allwords))

wfreq = FreqDist(allwords)
top1000 = wfreq.most_common(1000)

training = []
traininglabel = []

for p in poswords:
    vec = []
    for t in top1000:
        if t[0] in p:
            vec.append(1)
        else:
            vec.append(0)
    training.append(vec)
    traininglabel.append(1)

for n in negwords:
    vec = []
    for t in top1000:
        if t[0] in n:
            vec.append(1)
        else:
            vec.append(0)
    training.append(vec)
    traininglabel.append(0)


print(len(traininglabel))
print(len(training[0]))
print(training[0])

testing = []
testinglabel = []

for _, row in test.iterrows():
    words = process_text(row['Text'])
    vec = []
    for t in top1000:
        if t[0] in words:
            vec.append(1)
        else:
            vec.append(0)
    testing.append(vec)
    testinglabel.append(row['Label'])

print(len(testing))
print(len(testing[0]))

clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(2, 2), random_state=1, max_iter=500)
clf.fit(training, traininglabel)
predicted = clf.predict(testing)
print("Accuracy of Multi-Layer Perceptron")
print(metrics.classification_report(testinglabel, predicted))
lr = LogisticRegression()
lr.fit(training, traininglabel)
predicted = lr.predict(testing)

print("Accuracy of Logistic Regression")
print(metrics.classification_report(testinglabel, predicted))

## Naive Bayes
nb = GaussianNB()
nb.fit(training, traininglabel)
predicted = nb.predict(testing)
print("Accuracy of Naive Bayes")
print(metrics.classification_report(testinglabel, predicted))

## KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(training, traininglabel)
predicted = knn.predict(testing)

print("Accuracy of KNN")
print(metrics.classification_report(testinglabel, predicted))
