Movie Genre Classification

Project Description
This project focuses on developing a machine learning model that can classify movies into different genres based on their textual descriptions.
Given a dataset of movie plot summaries, the task is to clean and preprocess the text, vectorize it, and train a classifier to predict genres accurately.

The model ultimately allows the user to input any movie description and instantly predict the most likely genre.

*Technologies Used:*

--Python

--Pandas

--Scikit-learn

--NLTK (Natural Language Toolkit)

--Matplotlib & Seaborn (for visualization)

--ipywidgets (for GUI input)

*Dataset Details*
The datasets used are:

--train_data.csv: Contains movie descriptions and their corresponding genres (for training).

--test_data.csv: Contains movie descriptions without genres (for prediction).

--test_data_solution.csv: Contains the correct genres for the test data (for evaluating accuracy).

--Source: Kaggle - Movie Genre Classification

 *Project Workflow*
1. Importing Libraries
Essential libraries are imported to handle data reading, text processing, machine learning model building, and visualizations.

2. Loading the Data
Training, testing, and testing solution datasets are loaded using Pandas.

Sample rows from training data are printed to confirm successful loading.

3. Data Cleaning
Text descriptions are cleaned by:

Lowercasing the text.

Removing links, mentions, and non-alphabet characters.

Removing punctuation.

Tokenizing words and removing common stopwords.

This is done using NLTK tools like LancasterStemmer, stopwords, and regex functions.

4. Label Encoding
The movie genres (text labels like "Comedy", "Drama") are converted into numbers using LabelEncoder from Scikit-learn.

The original 'GENRE' column is dropped after encoding.

5. Model Building
A Pipeline is created consisting of:

CountVectorizer: Converts text into numerical feature vectors.

MultinomialNB (Naive Bayes Classifier): Trains a model on these vectors.

6. Training the Model
The cleaned descriptions and encoded genres are used to train the model.

7. Testing the Model
The model predicts genres for the test dataset.

Predictions are compared to the true labels to calculate accuracy.

Accuracy is printed to evaluate model performance.

8. Example Predictions
Two manually written movie descriptions are passed through the trained model.

Their predicted genres are printed for demonstration.

9. User Input (Interactive GUI)
Using ipywidgets, a simple interface is created where users can:

Input any movie description.

Click a "PREDICT" button.

View the predicted genre.

*Model Performance*
The model’s accuracy is calculated based on the predictions for the test dataset and compared against the true genres.
Accuracy achieved is displayed in the console.

*How to Run the Project*
Clone or download the project files into your local system.

Open the folder in VS Code.

Install the required libraries if not already installed:

--bash
--Copy
--Edit
--pip install pandas scikit-learn nltk matplotlib seaborn ipywidgets
--Make sure the dataset files (train_data.csv, test_data.csv, test_data_solution.csv) are present in the same directory.

Run the Python file:

--bash
--Copy
--Edit
--python genre_classifier.py
--Enter any movie description into the input box and press the "PREDICT" button to get the genre.

 *Folder Structure*
 
Copy
Edit
movie_genre_classifier/

├── train_data.csv

├── test_data.csv

├── test_data_solution.csv

├── genre_classifier.py

├── README.md

*Key Highlights*
End-to-end machine learning pipeline built using simple techniques.

Text preprocessing with custom cleaning function.

User-interactive prediction GUI using ipywidgets.

Real dataset used for training and evaluation.

Modular and beginner-friendly codebase for easy understanding and extension.

*Acknowledgements*
Kaggle Dataset Contributor: Youssefelbadry10

Libraries: Scikit-learn, NLTK, Matplotlib, Seaborn, ipywidgets

*Note*
This project is ideal for beginners in Machine Learning and Natural Language Processing (NLP) who want hands-on experience with a real-world classification problem!
