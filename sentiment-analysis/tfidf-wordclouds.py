import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---- Load & clean data (mirrors data.py) ----
combined = pd.read_csv('./Combined_News_DJIA.csv')
djia = pd.read_csv('./upload_DJIA_table.csv')

combined['Date'] = pd.to_datetime(combined['Date'])
djia['Date'] = pd.to_datetime(djia['Date'])
djia['Movement'] = (djia['Close'] > djia['Open']).astype(int)
movement_map = dict(zip(djia['Date'], djia['Movement']))
combined['Movement'] = combined['Date'].map(movement_map)
combined.loc[combined['Label'] != combined['Movement'], 'Label'] = \
    combined.loc[combined['Label'] != combined['Movement'], 'Movement']

headline_cols = [f'Top{i}' for i in range(1, 26)]

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'that',
    'this', 'it', 'its', 'he', 'she', 'they', 'we', 'i', 'you', 'his',
    'her', 'their', 'our', 'your', 'my', 'not', 'no', 'up', 'out', 'us',
    'after', 'over', 'new', 'more', 'about', 'than', 'into', 'says', 'say',
    'said', 'just', 'also', 'amid', 'all', 'who', 'what', 'when', 'where',
    'how', 'which', 'while', 'against', 'through', 'between', 'within',
    'any', 'other', 'each', 'such', 'like', 'both', 'many', 'much',
    'before', 'during', 'per', 'since', 'still', 'whether', 'back',
    'off', 'set', 'get', 'got', 'let', 'put', 'take', 'make', 'made',
    'come', 'came', 'now', 'then', 'there', 'here', 'even', 'most',
    'some', 'well', 'down', 'again', 'around', 'b', 'x', 's', 'r',
    'one', 'two', 'three', 'four', 'five', 'first', 'second', 'third',
}


def clean_headline(text):
    text = str(text)
    text = re.sub(r"^b['\"]|['\"]$", '', text.strip())
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text.lower().strip()


# Build one document per day: concatenate all 25 cleaned headlines
docs = []
labels = []
for _, row in combined.iterrows():
    day_text = ' '.join(
        clean_headline(row[col]) for col in headline_cols if not pd.isna(row[col])
    )
    docs.append(day_text)
    labels.append(row['Label'])

labels = np.array(labels)

# TF-IDF across all days — IDF penalizes words common to many days (both up & down)
vectorizer = TfidfVectorizer(
    stop_words=list(STOPWORDS),
    min_df=2,       # ignore words appearing on only 1 day
    token_pattern=r'[a-zA-Z]{3,}',  # at least 3 chars
    sublinear_tf=True,
)
tfidf_matrix = vectorizer.fit_transform(docs)   # shape: (n_days, n_vocab)
vocab = np.array(vectorizer.get_feature_names_out())

# Average TF-IDF per class (normalise for class size)
n_up   = (labels == 1).sum()
n_down = (labels == 0).sum()
up_avg   = np.asarray(tfidf_matrix[labels == 1].sum(axis=0)).flatten() / n_up
down_avg = np.asarray(tfidf_matrix[labels == 0].sum(axis=0)).flatten() / n_down

# Discriminative score: difference in class averages.
# Words common to both classes cancel out; only class-distinctive words survive.
up_diff   = up_avg   - down_avg   # positive → more prevalent on up days
down_diff = down_avg - up_avg     # positive → more prevalent on down days

up_freq   = {vocab[i]: float(up_diff[i])   for i in range(len(vocab)) if up_diff[i]   > 0}
down_freq = {vocab[i]: float(down_diff[i]) for i in range(len(vocab)) if down_diff[i] > 0}

print(f"Vocabulary size: {len(vocab)}")
print("Top 10 up-day words:  ", sorted(up_freq,   key=up_freq.get,   reverse=True)[:10])
print("Top 10 down-day words:", sorted(down_freq, key=down_freq.get, reverse=True)[:10])

# ---- Generate word clouds ----
wc_kwargs = dict(width=900, height=500, background_color='white',
                 max_words=150, collocations=False)

wc_up   = WordCloud(colormap='Greens', **wc_kwargs).generate_from_frequencies(up_freq)
wc_down = WordCloud(colormap='Reds',   **wc_kwargs).generate_from_frequencies(down_freq)

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.08)

ax_up   = fig.add_subplot(gs[0, 0])
ax_down = fig.add_subplot(gs[0, 1])

ax_up.imshow(wc_up, interpolation='bilinear')
ax_up.axis('off')
ax_up.set_title('DJIA Up Days (Label = 1)', fontsize=16, fontweight='bold',
                color='darkgreen', pad=12)

ax_down.imshow(wc_down, interpolation='bilinear')
ax_down.axis('off')
ax_down.set_title('DJIA Down Days (Label = 0)', fontsize=16, fontweight='bold',
                  color='darkred', pad=12)

fig.suptitle('TF-IDF Word Clouds — Words common to both classes are downweighted',
             fontsize=14, y=1.01)

out_path = './tfidf-wordclouds.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
plt.show()
