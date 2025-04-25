# Movie Genre Classification

## 🎯 Project Objective

The goal of this project is to build a machine learning model capable of classifying movies into their correct genres based solely on the textual descriptions of their plots.

### 🔍 Key Requirements:
- Process text-based movie plot descriptions into a clean, structured format.
- Apply text preprocessing techniques such as tokenization, stopword removal, and stemming.
- Convert cleaned text into numerical features using vectorization (e.g., CountVectorizer).
- Explore and evaluate multiple classification algorithms (e.g., Naive Bayes) to determine the most effective model.
- Train the model using a labeled dataset (with genres) and test its performance using a separate test dataset.

### ✅ Expected Outcome:
A well-trained and accurate classifier that can predict the genre of a movie based on its plot summary. This tool can assist movie platforms or users in auto-tagging movies by genre using only the plot description.

## 🛠️ Technologies Used

- **Python**: Main programming language used for data manipulation, model training, and evaluation.
- **Scikit-learn**: Library used for implementing machine learning algorithms.
- **NLTK**: Natural language processing library for tokenization, stopword removal, and stemming.
- **Pandas**: Library for data manipulation and cleaning.
- **Matplotlib / Seaborn**: Visualization libraries for performance evaluation.
- **Jupyter Notebook/VS Code**: Development environments used for coding.

## 📝 Project Description

In this project, we aim to classify movies into various genres (e.g., Drama, Thriller, Comedy) based on their plot descriptions. The following steps were undertaken to build the model:

### 1. Data Preprocessing
- **Text cleaning**: The movie descriptions are cleaned by removing URLs, special characters, and stopwords.
- **Tokenization**: The descriptions are tokenized into words to prepare them for vectorization.
- **Stemming**: Words are reduced to their root form to improve the quality of input to the model.

### 2. Model Training
We used **Multinomial Naive Bayes** as the classifier, which is suitable for text classification tasks. The cleaned descriptions are vectorized using **CountVectorizer** to convert text into numerical features.

### 3. Model Evaluation
The model was tested using a separate test dataset, and its performance was evaluated based on accuracy.

## 🧩 Code Structure

- **data_cleaning.py**: Contains functions for cleaning and preprocessing the data.
- **model_training.py**: Code for training the model using a Naive Bayes classifier and testing it on the test data.
- **predict_genre.py**: Code that allows users to input movie descriptions and predict their genres using the trained model.
- **requirements.txt**: Lists all the Python libraries required to run the project.

## 📥 Dataset

The dataset used in this project is the **Movie Genre Classification** dataset from Kaggle. It contains movie titles, descriptions, and their corresponding genres. You can find the dataset [here](https://www.kaggle.com/code/youssefelbadry10/movie-genre-classification/input).

## ⚙️ How to Run the Code

### Step 1: Install Dependencies
You will need Python 3.x installed along with the following libraries:
```bash
pip install -r requirements.txt

### Step 2: Load the Dataset
Download the dataset from the provided link and save the CSV files in your project directory.

### Step 3: Run the Model
You can train the model by running the following command:
bash
Copy
Edit
python model_training.py

### Step 4: Make Predictions
Once the model is trained, you can use it to predict the genre of a movie by providing a movie description. You can run the following code to predict a genre:
bash
Copy
Edit
python predict_genre.py

### Step 5: Evaluate the Model
The model's accuracy will be printed to the console after training.

🔬 Model Performance
The current model uses the Multinomial Naive Bayes algorithm, achieving an accuracy of 53.75% on the test set.

📊 Results:
Accuracy: 53.75%

The model performs well, but future improvements can be made by exploring other algorithms, vectorization methods, and hyperparameter tuning.

📝 How to Contribute
If you would like to contribute to this project, feel free to fork the repository, create a new branch, and submit a pull request. All contributions are welcome!

📄 License
This project is licensed under the MIT License.
markdown
Copy
Edit

### Key Points Covered:
- **Project Objective**: Clearly states the goal and what is expected.
- **Technologies Used**: Lists the main libraries and tools you used.
- **Project Description**: Explains the steps involved in data preprocessing, model training, and evaluation.
- **How to Run the Code**: Step-by-step instructions for setting up and running the code, including installing dependencies and running Python scripts.
- **Model Performance**: Mentions the achieved accuracy and encourages future improvements.
