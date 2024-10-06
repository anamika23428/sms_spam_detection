# SMS Spam Detection

This project aims to detect spam messages using a machine learning model trained on SMS data. The Naive Bayes algorithm was chosen for model training as it provided the highest accuracy and precision.

## Dataset
The dataset contains SMS messages labeled as either "spam" or "ham" (non-spam). It is processed and transformed to a suitable format for model training.

## Model
The Naive Bayes algorithm was used due to its effectiveness in text classification tasks. After comparing different approaches, Naive Bayes provided the best results in terms of both accuracy and precision.
## Key Libraries

- **pandas**: For data manipulation and processing.
  
- **sklearn**: A machine learning library used for model training and evaluation.
  - `LabelEncoder`: Converts categorical target labels ("spam" and "ham") into numerical values (1 and 0) to make the data suitable for training.
  - `CountVectorizer`: Converts the text data into a matrix of token counts, used to transform the SMS messages into a format that the machine learning model can process.
  - `TfidfVectorizer`: Tf-idf is one of the best metrics to determine how significant a term is to a text in a series or a corpus. tf-idf is a weighting system that assigns a weight to each word in a document based on its term frequency (tf) and the reciprocal document frequency (tf) (idf). The words with higher scores of weight are deemed to be more significant.

- **nltk** (Natural Language Toolkit): A library for text processing and cleaning.
  - `StopWords`: A list of commonly used words (like "the", "and", "is") that are removed from the dataset to reduce noise in the data.
  - `PorterStemmer`: A tool used to reduce words to their root form (e.g., "running" to "run") for more effective analysis.

- **Streamlit**: Used to build a simple web interface to classify SMS messages as spam or ham in real-time.
- **pickle**: For saving and loading the trained model. The trained Naive Bayes model is serialized using `pickle` to make it reusable for future predictions without retraining.

## Workflow
1. **Data Preprocessing**:
   - SMS messages are cleaned and prepared for model training.
   - Categorical labels ("spam" and "ham") are encoded as numerical values using `LabelEncoder` from `sklearn`.
   - Text data is vectorized using `TfidfVectorizer` for input into the model.

2. **Model Training**:
   - A Naive Bayes classifier is trained on the processed data.
   - The model's performance is evaluated using accuracy and precision metrics.

3. **Deployment**:
   - The classifier is deployed using Streamlit to create a user-friendly interface for classifying SMS messages.

## Installation
To run this project locally, install the required dependencies:

```bash
pip install pandas scikit-learn streamlit
Make sure to also download the NLTK stopwords:
python -m nltk.downloader stopwords
```
## Run the Streamlit app: 
Launch the Streamlit app by running the following command:
streamlit run app.py

  
