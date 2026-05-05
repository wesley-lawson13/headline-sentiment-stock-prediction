import re
from collections import Counter
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data import train, test

# ---- TEXT PROCESSING (mirrors bag-of-words.py) ----

nltk.download('stopwords', quiet=True)
stops = set(stopwords.words('english'))
stops.update([",", ".", "!", "?", "'", '"', "i", "n't", "'ve", "'d", "'s", "says", "new", "people", "one"])

def process_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return list(set(w for w in words if w not in stops))

# ---- BUILD BOW VOCABULARY FROM TRAINING DATA ----

poswords = []   # list of word-sets, one per positive training day
negwords = []   # list of word-sets, one per negative training day
allwords = []   # flat list for FreqDist

for _, row in train.iterrows():
    words = process_text(row['Text'])
    allwords.extend(words)
    if row['Label'] == 1:
        poswords.append(words)
    else:
        negwords.append(words)

wfreq = FreqDist(allwords)
top1000 = wfreq.most_common(1000)
vocab = [w for w, _ in top1000]
vocab_set = set(vocab)

# ---- VECTORIZE ----

def vectorize(word_list):
    return [1 if w in word_list else 0 for w in vocab]

training = [vectorize(p) for p in poswords] + [vectorize(n) for n in negwords]
traininglabel = [1] * len(poswords) + [0] * len(negwords)

testing = []
testinglabel = []
for _, row in test.iterrows():
    words = process_text(row['Text'])
    testing.append(vectorize(words))
    testinglabel.append(row['Label'])

# ---- KNN MODEL ----

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(training, traininglabel)
predicted = knn.predict(testing)

print("BOW + KNN Classification Report")
print(metrics.classification_report(testinglabel, predicted))

# ---- WORD FREQUENCIES FOR CLOUDS ----
# Weight = number of training documents (days) in which the word appears,
# restricted to the top-1000 BOW vocabulary.

pos_freq = Counter()
neg_freq = Counter()
all_freq = Counter()

for word_list in poswords:
    for w in word_list:
        if w in vocab_set:
            pos_freq[w] += 1
            all_freq[w] += 1

for word_list in negwords:
    for w in word_list:
        if w in vocab_set:
            neg_freq[w] += 1
            all_freq[w] += 1

# ---- GENERATE WORD CLOUDS ----

wc_kwargs = dict(
    width=900, height=500,
    background_color='white',
    max_words=150,
    collocations=False,
)

wc_pos = WordCloud(colormap='Greens', **wc_kwargs).generate_from_frequencies(pos_freq)
wc_neg = WordCloud(colormap='Reds',   **wc_kwargs).generate_from_frequencies(neg_freq)
wc_all = WordCloud(colormap='Blues',  **wc_kwargs).generate_from_frequencies(all_freq)

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.1)

ax_pos = fig.add_subplot(gs[0, 0])
ax_neg = fig.add_subplot(gs[0, 1])
ax_all = fig.add_subplot(gs[1, :])

ax_pos.imshow(wc_pos, interpolation='bilinear')
ax_pos.axis('off')
ax_pos.set_title('DJIA Up Days (Positive)', fontsize=16, fontweight='bold', color='darkgreen', pad=12)

ax_neg.imshow(wc_neg, interpolation='bilinear')
ax_neg.axis('off')
ax_neg.set_title('DJIA Down Days (Negative)', fontsize=16, fontweight='bold', color='darkred', pad=12)

ax_all.imshow(wc_all, interpolation='bilinear')
ax_all.axis('off')
ax_all.set_title('All Training Days', fontsize=16, fontweight='bold', color='steelblue', pad=12)

fig.suptitle('BOW + KNN — Word Clouds from Model Vocabulary', fontsize=20, fontweight='bold', y=0.98)

out_path = './bow-knn-wordclouds.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
plt.show()
