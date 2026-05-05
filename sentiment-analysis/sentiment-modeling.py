# TODO
# -> Maybe apply FinBERT?
# -> later... create a word cloud?

import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

from data import train, test

# ---- HEADLINE CLEANING ----

def clean_headline(text):
    """Strip Python byte-string literal formatting (e.g. b'...') from raw headlines."""
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, bytes):
            return parsed.decode('utf-8', errors='ignore')
        return str(parsed)
    except Exception:
        return str(text)

HEADLINE_COLS = [f'Top{i}' for i in range(1, 26)]

# ---- VADER FEATURE EXTRACTION ----

analyzer = SentimentIntensityAnalyzer()

def extract_features(row):
    """
    Score each headline individually with VADER and aggregate per day.
    Features:
      - mean/std/min/max compound score across all headlines
      - mean positive and negative scores
      - fraction of headlines that are positive (compound > 0.05)
      - fraction of headlines that are negative (compound < -0.05)
    neu_mean omitted — it's ~(1 - pos_mean - neg_mean) and adds no signal.
    compound_std captures headline consensus vs. disagreement.
    """
    scores = []
    for col in HEADLINE_COLS:
        text = row[col]
        if not isinstance(text, str) or text.strip() == '':
            continue
        cleaned = clean_headline(text)
        scores.append(analyzer.polarity_scores(cleaned))

    if not scores:
        return {
            'compound_mean': 0, 'compound_std': 0, 'compound_min': 0, 'compound_max': 0,
            'pos_mean': 0, 'neg_mean': 0,
            'frac_positive': 0, 'frac_negative': 0,
        }

    compounds = [s['compound'] for s in scores]
    return {
        'compound_mean': np.mean(compounds),
        'compound_std':  np.std(compounds),
        'compound_min':  np.min(compounds),
        'compound_max':  np.max(compounds),
        'pos_mean':      np.mean([s['pos'] for s in scores]),
        'neg_mean':      np.mean([s['neg'] for s in scores]),
        'frac_positive': np.mean([c > 0.05  for c in compounds]),
        'frac_negative': np.mean([c < -0.05 for c in compounds]),
    }

print("Extracting VADER features...")
train_features = train.apply(extract_features, axis=1, result_type='expand')
test_features  = test.apply(extract_features,  axis=1, result_type='expand')

feature_cols = train_features.columns.tolist()
X_train, y_train = train_features[feature_cols].values, train['Label'].values
X_test,  y_test  = test_features[feature_cols].values,  test['Label'].values

# ---- MODELS ----

# Logistic Regression (scaled)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))

# Random Forest (no scaling needed)
rf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# ---- CNN ----

class SentimentCNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, n_features) for Conv1d
        x = self.conv(x)
        return self.classifier(x).squeeze(1)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

cnn = SentimentCNN(n_features=len(feature_cols))
optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

cnn.train()
for epoch in range(50):
    for xb, yb in train_loader:
        optimizer.zero_grad()
        criterion(cnn(xb), yb).backward()
        optimizer.step()

cnn.eval()
with torch.no_grad():
    logits = cnn(X_test_t)
    cnn_preds = (torch.sigmoid(logits) >= 0.5).long().numpy()
cnn_acc = accuracy_score(y_test, cnn_preds)

# ---- RESULTS ----

LABELS = {
    'compound_mean': 'Avg. Sentiment Score',
    'compound_std':  'Sentiment Spread (Std Dev)',
    'compound_min':  'Most Negative Headline',
    'compound_max':  'Most Positive Headline',
    'pos_mean':      'Avg. Positivity',
    'neg_mean':      'Avg. Negativity',
    'frac_positive': 'Share of Positive Headlines',
    'frac_negative': 'Share of Negative Headlines',
}

# Example daily descriptor (first test day)
example_date = test.iloc[0]['Date']
example_label = 'Up' if test.iloc[0]['Label'] == 1 else 'Down'
example_features = test_features.iloc[0]

feat_table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", padding=(0, 1))
feat_table.add_column("Feature", style="bold white", min_width=32)
feat_table.add_column("Value", justify="right", min_width=10)

for col, label in LABELS.items():
    val = example_features[col]
    if col in ('frac_positive', 'frac_negative'):
        formatted = f"{val * 100:.1f}%"
        color = "green" if col == 'frac_positive' else "red"
    else:
        formatted = f"{val:+.4f}"
        color = "green" if val > 0 else ("red" if val < 0 else "white")
    feat_table.add_row(label, f"[{color}]{formatted}[/{color}]")

direction_color = "green" if example_label == "Up" else "red"
console.print()
console.print(Panel(
    feat_table,
    title=f"[bold]VADER Features  ·  {example_date}   DJIA: [{direction_color}]{example_label}[/{direction_color}][/bold]",
    border_style="bright_blue",
    padding=(1, 2),
))

# Model performance comparison
model_table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", padding=(0, 1))
model_table.add_column("Model", style="bold white", min_width=25)
model_table.add_column("Accuracy", justify="right", min_width=10)

best_acc = max(lr_acc, rf_acc, cnn_acc)
for name, acc in [("Logistic Regression", lr_acc), ("Random Forest", rf_acc), ("CNN", cnn_acc)]:
    star = " ★" if acc == best_acc else ""
    color = "bright_green" if acc == best_acc else "white"
    model_table.add_row(f"[{color}]{name}{star}[/{color}]", f"[{color}]{acc * 100:.1f}%[/{color}]")

console.print(Panel(
    model_table,
    title="[bold]Model Performance Comparison[/bold]",
    border_style="bright_blue",
    padding=(1, 2),
))
