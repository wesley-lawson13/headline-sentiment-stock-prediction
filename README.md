# headline-sentiment-stock-prediction
Boston College CSCI3349 Final Project: Group 6. Analyzes the sentiment of financial headlines to predict whether the DJIA went up or down on a given day.

___

## Directory Structure

- `README.md`: The README file for this repository.
- `Combined_News_DJIA.csv` and `upload_DJIA_table.csv`: The imported datasets we used for this project (a copy of each is also in the sentiment-analysis/ folder).
- `DJIA_NLP_Prediction.ipynb`: The Colab notebook that contains all of our code for this project organized into sections.
- `model_results_*.csv` files: The generated output for each model. Generated from the DJIA_NLP_Prediction.ipynb file.
- `Results Visualization.ipynb`: The Colab notebook with most of the results visualizations.
- `sentiment-analysis/`: Folder containing the generated word clouds (bag-of-words, tf-idf), and the VADER sentiment analysis. The code for the generated baselines, data prep, and the VADER sentiment was also included in the DJIA_NLP_Prediction.ipynb file.
    - `sentiment-modeling.py`: the VADER sentiment analysis file that shows the output of the VADER sentiment scores and descriptors as features for three separate models (logisitic regression, random forest, and a CNN). It also shows the VADER sentiment output for an example day.
    - .py wordcloud files: Generates word clouds based on a bag-of-words (bag-of-words.py) + KNN model, as well as the tf-idf normalized wordclouds. 
    - .png wordcloud files: The output of the python wordcloud scripts.
- `results/` and `results_no_2008/`: Folders containing training data for the BERT models. Included to allow BERT to not have to retrain itself each time the DJIA_NLP_Prediction.ipynb is ran. Omitted because the folders were too large.

___ 

## Executing Files in the sentiment-analysis/ Directory 

To run files in this directory:

1. Create a Virtual Environment
`python -m venv venv`

2. Activate the Virtual Environment
`source venv/bin/activate`

3. Install Requirements
`pip install -r requirements.txt`

4. Run the code files
`python file.py`

When you run the `sentiment-modeling.py` file and the word cloud files, it will prep the data and run the necessary models, so you don't have to run any other files before running the files you are interested in looking at.
___

## Running the DJIA_NLP_Prediction.ipynb file

To run this notebook:

1. Ensure you have the necessary Google Colab permissions and drive access configured.
2. Execute the 'Packages' section to install all required libraries.
3. Run the 'Data Preprocessing' section to load and clean the dataset. This is where you would decide to include / not include the 2008 data as well as set up the output paths to show the results in the generated CSV files.
4. Proceed through the 'Generate Baselines', 'Bag of Words', 'Word2Vec', 'BERT', 'Mistral', and 'VADER Sentiment Analysis' sections to train and evaluate different models, or do each individually.
5. Review the results in the generated CSV files in your Google Drive.

___

## Data 

We found our datasets on Kaggle. Here is a link to the Kaggle page: [Datasets](https://www.kaggle.com/datasets/tanishqdublish/stock-market-predictions/data)

___ 

## Non-Standard Libraries / APIs

Below are the non-standard libraries we used in this project. For a more detailed list, see the `DJIA_NLP_Prediction.ipynb` file.

- Mistral: The Python Mistral client we used for zero shot, one shot, and three shot learning attempts. [mistralai](https://github.com/mistralai/client-python)
- VADER: A sentiment analysis tool tuned for social media / news texts. [vaderSentiment](https://github.com/cjhutto/vaderSentiment)

Additionally, here are all the libraries used in the `DJIA_NLP_Prediction.ipynb` file:

*   **Data Handling and Analysis:**
    *   `pandas` (pd)
    *   `numpy` (np)
    *   `datasets` (from Hugging Face)

*   **Natural Language Processing (NLP):**
    *   `nltk` (Natural Language Toolkit)
    *   `gensim` (Word2Vec)
    *   `vaderSentiment` (Sentiment Intensity Analyzer)
    *   `transformers` (from Hugging Face)
    *   `mistralai` (Mistral API client)

*   **Machine Learning / Deep Learning Frameworks and Libraries:**
    *   `tensorflow` (tf)
    *   `sklearn` (Scikit-learn) - including `metrics`, `ensemble`, `linear_model`, `naive_bayes`, `neighbors`, `neural_network`, `preprocessing`, `svm`
    *   `keras` (within TensorFlow for `Sequential` model and layers like `Conv1D`, `MaxPooling1D`, `Dropout`, `Dense`, `Flatten`)

*   **Evaluation Metrics:**
    *   `evaluate` (from Hugging Face, for metrics like accuracy and F1-score)

*   **Utilities:**
    *   `re` (Regular Expressions)
    *   `math`
    *   `time`
    *   `ast`
    *   `os`

*   **Google Colab Specific:**
    *   `google.colab` (`drive`)

*   **Other:**
    *   `glob` (for pathnames matching a specified pattern - though not directly used, it's imported)

___
