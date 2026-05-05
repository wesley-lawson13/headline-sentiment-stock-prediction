import pandas as pd

combined = pd.read_csv('./Combined_News_DJIA.csv')
djia = pd.read_csv('./upload_DJIA_table.csv')

# ---- DATA CLEANING / PREP ---- 

combined['Date'] = pd.to_datetime(combined['Date'])
djia['Date'] = pd.to_datetime(djia['Date'])
djia['Movement'] = (djia['Close'] > djia['Open']).astype(int)
movement_map = dict(zip(djia['Date'], djia['Movement']))
combined['Movement'] = combined['Date'].map(movement_map)
combined['Match'] = combined['Label'] == combined['Movement']
combined['Match'] = combined['Label'] == combined['Movement']
accuracy_before = combined['Match'].mean()
print(combined.loc[~combined['Match'], ['Date', 'Label', 'Movement']])
combined.loc[~combined['Match'], 'Label'] = combined.loc[~combined['Match'], 'Movement']
combined['Match'] = combined['Label'] == combined['Movement']
accuracy_after = combined['Match'].mean()
print("Accuracy before:", accuracy_before)
print("Accuracy after:", accuracy_after)

# Combine into one text column
headline_cols = [f'Top{i}' for i in range(1, 26)]
combined['Text'] = combined[headline_cols].fillna('').agg(' '.join, axis=1)

#split into test and training
train = combined.sample(n=1600, random_state=42)
test = combined.drop(train.index)
print("Train size:", len(train))
print("Test size:", len(test))
