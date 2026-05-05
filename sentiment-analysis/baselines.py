from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

from data import train, test, combined

#Majority Baseline
majority_label = train['Label'].mode()[0]
majority_predictions = [majority_label] * len(test)
majority_accuracy = accuracy_score(test['Label'], majority_predictions)

#Random Baseline
np.random.seed(42)
random_predictions = np.random.randint(0, 2, size=len(test))
random_accuracy = accuracy_score(test['Label'], random_predictions)
sorted_combined = combined.sort_values(by='Date')
sorted_combined['Previous_Day_Label'] = sorted_combined['Label'].shift(1)

#Momentum Baseline
prev_day_label_map = sorted_combined.set_index('Date')['Previous_Day_Label'].dropna().to_dict()
test_with_prev_label = test.sort_values(by='Date').copy()
test_with_prev_label = pd.merge(test_with_prev_label, sorted_combined[['Date', 'Previous_Day_Label']], on='Date', how='left')
test_with_prev_label['Previous_Day_Label'] = test_with_prev_label['Previous_Day_Label'].fillna(majority_label)
previous_day_predictions = test_with_prev_label['Previous_Day_Label'].astype(int)
previous_day_accuracy = accuracy_score(test_with_prev_label['Label'], previous_day_predictions)

if __name__ == '__main__':
    print(f"Majority Baseline Accuracy: {majority_accuracy:.4f}")
    print(f"Random Baseline Accuracy: {random_accuracy:.4f}")
    print(f"Previous Day Momentum Baseline Accuracy: {previous_day_accuracy:.4f}")
