# headline-sentiment-stock-prediction
Boston College CSCI3349 Final Project: Group 6. Analyzes the sentiment of financial headlines to predict whether the DJIA went up or down on a given day.

___

## Folder Structure

- README.md: The README file for this repository.
- Combined_News_DJIA.csv and upload_DJIA_table.csv: The imported datasets we used for this project (a copy of each is also in the sentiment-analysis/ folder).
- DJIA_NLP_Prediction.ipynb: The Colab notebook that contains all of our code for this project organized into sections.
- model_results_*.csv files: The generated output for each model. Generated from the DJIA_NLP_Prediction.ipynb file.
- Results Visualization.ipynb: The Colab notebook with most of the results visualizations.
- sentiment-analysis/: Folder containing the generated word clouds (bag-of-words, tf-idf), and the VADER sentiment analysis. The code for the generated baselines, data prep, and the VADER sentiment was also included in the DJIA_NLP_Prediction.ipynb file.
    - sentiment-modeling.py: the VADER sentiment analysis file that shows the output of the VADER sentiment scores and descriptors as features for three separate models (logisitic regression, random forest, and a CNN). It also shows the VADER sentiment output for an example day.
    - .py wordcloud files: Generates word clouds based on a bag-of-words (bag-of-words.py) + KNN model, as well as the tf-idf normalized wordclouds. 
    - .png wordcloud files: The output of the python wordcloud scripts.

___ 

## Executing Files in the sentiment-analysis/ Folder

1. Create a Virtual Environment
`python -m venv venv`

2. Activate the Virtual Environment
`source venv/bin/activate`

3. Install Requirements
`pip install -r requirements.txt`

4. Run the code files
`python file.py`

___

### Running the DJIA_NLP_Prediction.ipynb file

To run this notebook:

1. Ensure you have the necessary Google Colab permissions and drive access configured.
2. Execute the 'Packages' section to install all required libraries.
3. Run the 'Data Preprocessing' section to load and clean the dataset. This is where you would decide to include / not include the 2008 data as well as set up the output paths to demonstrate the results in generated CSV files.
4. Proceed through the 'Generate Baselines', 'Bag of Words', 'Word2Vec', 'BERT', 'Mistral', and 'VADER Sentiment Analysis' sections to train and evaluate different models, or do each individually.
5. Review the results in the generated CSV files in your Google Drive.
