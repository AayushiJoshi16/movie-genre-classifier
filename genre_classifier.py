# Importing Libraries
import pandas as pd  # Reading Data
from sklearn.naive_bayes import MultinomialNB  # The Prediction Model
from sklearn.feature_extraction.text import CountVectorizer  # The transforming To Vector Tool
from sklearn.pipeline import Pipeline  # Running Line by line Tool
from sklearn.model_selection import train_test_split  # Splitting Tool
from sklearn.metrics import confusion_matrix, accuracy_score  # Visualization
import matplotlib.pyplot as plt  # Visualization
import seaborn as sn  # Visualization
import warnings

warnings.filterwarnings("ignore")

# Reading the Data (Assume that dataset is in the same directory or specify the full path)
TrainData = pd.read_csv("train_data.csv")  # Update with the correct path
TestData = pd.read_csv("test_data.csv")  # Update with the correct path
TestDataSol = pd.read_csv("test_data_solution.csv")  # Update with the correct path

# Display first few rows of the training data to confirm loading
print("Train Data Loaded Successfully!")
print(TrainData.head())

# Cleaning The Data
import nltk
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
import re
import string

# Initialize stemmer and stopwords
stemmer = LancasterStemmer()
stop_words = set(stopwords.words("english"))

def cleaning_data(text):
    text = text.lower()
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'.pic\S+', '', text)
    text = re.sub(r'[^a-zA-Z+]', ' ', text)
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    text = " ".join([i for i in words if i not in stop_words and len(i) > 2])
    text = re.sub(r"\s+", " ", text).strip()
    return text

TrainData["TextCleaning"] = TrainData["DESCRIPTION"].apply(cleaning_data)
TestData["TextCleaning"] = TestData["DESCRIPTION"].apply(cleaning_data)

# Confirm data cleaning
print("Data Cleaning Completed!")

# Encoding the Labels (Genres)
from sklearn.preprocessing import LabelEncoder

GENRElabel = LabelEncoder()
TrainData['GENRE_n'] = GENRElabel.fit_transform(TrainData['GENRE'])

# Dropping old column
TrainData = TrainData.drop("GENRE", axis=1)

# Print number of unique genres
print("Number of Unique Genres: ", len(TrainData.GENRE_n.unique()))

# Creating Model Pipeline
clf = Pipeline([('vectorizer', CountVectorizer()), ('nb', MultinomialNB())])

# Training the model
print("Training the Model...")
clf.fit(TrainData.TextCleaning, TrainData.GENRE_n)
print("Model Training Completed!")

# Making predictions on the test data
y_pred = clf.predict(TestData.TextCleaning)
y_true = GENRElabel.fit_transform(TestDataSol['GENRE'])

# Model Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Example prediction (testing manual input)
Descriptions = [
    "Listening in to a conversation between his doctor and parents, 10-year-old Oscar learns what nobody has the courage to tell him. He only has a few weeks to live. Furious, he refuses to speak to anyone except straight-talking Rose, the lady in pink he meets on the hospital stairs. As Christmas approaches, Rose uses her fantastical experiences as a professional wrestler, her imagination, wit and charm to allow Oscar to live life and love to the full, in the company of his friends Pop Corn, Einstein, Bacon and childhood sweetheart Peggy Blue.",
    "In tough economic times Max and Joey have all but run out of ideas until, they discover that senior housing is cheap. Not only that but Max's aunt just kicked the bucket and no one knows yet. In a hilarious series that always keeps you on your toes, the two friends take us on a cross-dressing, desperate and endearing ride through being broke."
]
print("Predicted Genre for Example Descriptions:", GENRElabel.inverse_transform(clf.predict(Descriptions)))

# GUI for user input
import ipywidgets as widgets
from IPython.display import display

# Define a function to be called when the button is clicked
def on_button_click(b):
    text_value = text_box.value
    print("Film's Genre is: ", GENRElabel.inverse_transform([int(clf.predict([text_value]))]))

# Create button and text box widgets
button = widgets.Button(description="PREDICT")
text_box = widgets.Text(placeholder="Enter description")
text_box.layout.width = '500px'
text_box.layout.height = '30px'

# Attach the function to the button click event
button.on_click(on_button_click)

# Display the widgets
display(text_box)
display(button)
