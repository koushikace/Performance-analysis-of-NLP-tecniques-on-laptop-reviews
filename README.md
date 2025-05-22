Performance Analysis of NLP Techniques on Laptop Reviews
Project Overview
This project focuses on performing sentiment analysis on a dataset of Amazon laptop reviews. The primary goal is to classify reviews as either "positive" or "negative/neutral" based on their text content and to evaluate the performance of various Natural Language Processing (NLP) techniques and machine learning models in achieving this classification.

The process involves extensive data preprocessing, including text cleaning and normalization, feature extraction, handling class imbalance, and training and evaluating different classification algorithms.

Dataset
The dataset used for this analysis is Cleansed_laptop data.csv, which contains Amazon laptop reviews.
Key columns include:

comment: The text content of the review.

rating: The numerical rating given by the reviewer (1-5 stars).

For the purpose of this sentiment analysis, the rating column was transformed into a binary target variable:

Ratings 1, 2, and 3 are classified as 0 (Negative/Neutral).

Ratings 4 and 5 are classified as 1 (Positive).

Key Features and Techniques Used
This project demonstrates a comprehensive NLP and machine learning pipeline, incorporating the following techniques:

Data Loading & Initial Exploration: Using pandas to load and inspect the dataset.

Text Preprocessing:

Lowercasing: Converting all text to lowercase to ensure consistency.

Punctuation and Special Character Removal: Cleaning text by removing non-alphabetic characters.

Stop Word Removal: Eliminating common words (e.g., "the", "is", "a") that do not carry significant sentiment.

Lemmatization: Reducing words to their base or root form (e.g., "running" to "run") using NLTK's WordNetLemmatizer.

Feature Extraction:

TF-IDF (Term Frequency-Inverse Document Frequency): Converting cleaned text data into numerical feature vectors that reflect the importance of words in a document relative to the entire corpus.

Handling Class Imbalance:

SMOTE (Synthetic Minority Over-sampling Technique): Generating synthetic samples for the minority class (positive reviews) to balance the dataset, preventing models from being biased towards the majority class.

Model Training and Evaluation:

Data Splitting: Dividing the dataset into training, validation, and testing sets to ensure robust model evaluation.

Decision Tree Classifier: Training and evaluating a Decision Tree model as a baseline.

Performance Metrics: Calculating and visualizing:

Confusion Matrix: To understand true positives, true negatives, false positives, and false negatives.

Precision: The accuracy of positive predictions.

Recall: The ability of the model to find all positive samples.

Accuracy: The overall correctness of the model.

F1-Score: The harmonic mean of precision and recall, providing a balanced measure.

(Future work includes evaluating other models like Random Forest, AdaBoost, and Gradient Boosting.)

Results (Decision Tree Model)
After preprocessing, TF-IDF vectorization, and SMOTE oversampling, the Decision Tree Classifier achieved the following performance on the validation set:

Precision: 80.43%

Recall: 80.43%

Accuracy: 80.43%

F1-Score: 80.43%

The consistent scores across precision, recall, accuracy, and F1-score indicate a balanced performance, likely attributed to the effective handling of class imbalance using SMOTE.

Repository Structure
.
├── Cleansed_laptop data.csv      # The raw dataset of laptop reviews
├── Performance_Analysis_NLP.ipynb # Jupyter Notebook containing the analysis code
├── README.md                     # This file
└── requirements.txt              # List of Python dependencies

How to Run the Project Locally
To set up and run this project on your local machine, follow these steps:

Clone the Repository:

git clone https://github.com/koushikace/Performance-analysis-of-NLP-tecniques-on-laptop-reviews.git

Navigate to the Project Directory:

cd Performance-analysis-of-NLP-tecniques-on-laptop-reviews

Create a Virtual Environment (Recommended):

python -m venv venv

Activate the Virtual Environment:

Windows:

.\venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

Install Dependencies:

pip install -r requirements.txt

(If requirements.txt is not present, you'll need to install the libraries manually: pandas, scikit-learn, imblearn, nltk, regex.)

pip install pandas scikit-learn imblearn nltk regex

Download NLTK Data:
Open a Python interpreter or add the following lines to the top of your notebook and run them:

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

Open the Jupyter Notebook:

jupyter notebook Performance_Analysis_NLP.ipynb

This will open the notebook in your web browser, where you can run the cells and explore the analysis.

Future Work and Improvements
Explore More Models: Implement and compare the performance of other classification algorithms like Random Forest, AdaBoost, Gradient Boosting, Naive Bayes, and Support Vector Machines.

Hyperparameter Tuning: Optimize the parameters of the chosen models using techniques like GridSearchCV or RandomizedSearchCV for better performance.

Deep Learning Models: Experiment with neural networks (e.g., LSTMs, GRUs) for sentiment analysis, especially if the dataset size increases.

Advanced Text Embeddings: Investigate pre-trained word embeddings (e.g., Word2Vec, GloVe, FastText) or contextual embeddings (e.g., BERT, RoBERTa) for feature representation.

Error Analysis: Delve deeper into misclassified reviews to identify patterns and potential areas for improving preprocessing or model architecture.

Deployment: Consider deploying the trained model as a simple web application (e.g., using Flask or Streamlit) to allow users to input reviews and get real-time sentiment predictions.

Contact
If you have any questions or suggestions, feel free to reach out:

GitHub: koushikace

License
This project is licensed under the MIT License - see the LICENSE file for details.
