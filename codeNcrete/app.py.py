# Import all necessary libraries
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

### --- 1. LOAD AND PREPARE THE DATA --- ###
print("Starting the project...")
print("Step 1: Loading and preparing data...")

try:
    # Load the dataset you downloaded from Kaggle
    df = pd.read_csv('../csv/labeled_data.csv')
except FileNotFoundError:
    print("\n--- ERROR ---")
    print("File 'labeled_data.csv' not found.")
    print("Please download it from Kaggle and save it in the same folder as this script.")
    print("Kaggle URL: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset")
    exit()

# The original dataset has 3 classes:
# 0 = hate speech
# 1 = offensive language
# 2 = neither
#
# Let's simplify this. We'll create a new binary (two-class) label:
# 1 = Harmful (this will be for both hate speech and offensive language)
# 0 = Not Harmful (this will be for 'neither')

# The 'map' function applies this change.
df['label'] = df['class'].map({0: 1, 1: 1, 2: 0})

# Select only the columns we need: the text and our new label
df_clean = df[['tweet', 'label']]


### --- 2. CLEAN THE TEXT DATA --- ###
print("Step 2: Cleaning the text data...")

def clean_text(text):
    """
    A simple function to clean the text data.
    - Removes URLs
    - Removes user mentions (@username)
    - Removes special characters (leaves only letters and numbers)
    - Converts to lowercase
    """
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove user mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special characters
    text = text.lower().strip()          # Convert to lowercase and strip whitespace
    return text

# Apply the cleaning function to every tweet
df_clean['tweet'] = df_clean['tweet'].apply(clean_text)


### --- 3. DEFINE FEATURES (X) AND TARGET (y) --- ###

# X is the 'feature' - the text data we use to make predictions
X = df_clean['tweet']

# y is the 'target' - the label (0 or 1) we want to predict
y = df_clean['label']


### --- 4. SPLIT DATA INTO TRAINING AND TESTING SETS --- ###
print("Step 3: Splitting data into training and testing sets...")

# We use 80% of the data to train the model and 20% to test it
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20,  # 20% for testing
    random_state=42, # Ensures the split is the same every time
    stratify=y       # Ensures train/test sets have a similar balance of labels
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


### --- 5. VECTORIZE THE TEXT (TEXT-TO-NUMBERS) --- ###
print("Step 4: Turning text into numerical features (Vectorizing)...")

# A machine can't understand words. TfidfVectorizer converts
# text into a matrix of numbers based on word frequency.
vectorizer = TfidfVectorizer(
    stop_words='english', # Ignore common English words (like 'the', 'is', 'a')
    max_features=5000     # Only use the top 5000 most common words
)

# Learn the vocabulary from the training data
X_train_vec = vectorizer.fit_transform(X_train)

# Transform the test data using the *same* vocabulary
X_test_vec = vectorizer.transform(X_test)


### --- 6. TRAIN THE SUPERVISED LEARNING MODEL --- ###
print("Step 5: Training the Logistic Regression model...")

# Logistic Regression is a fast, reliable, and easy-to-interpret
# model for binary classification.
model = LogisticRegression(max_iter=1000)

# Train the model on the training data
model.fit(X_train_vec, y_train)

print("Model training complete!")


### --- 7. EVALUATE THE MODEL'S PERFORMANCE --- ###
print("\n--- MODEL EVALUATION RESULTS ---")

# Make predictions on the test data (data it's never seen)
y_pred = model.predict(X_test_vec)

# Compare the model's predictions (y_pred) to the true labels (y_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

# Print a detailed report (precision, recall, f1-score)
# Precision: Of the posts we *predicted* as Harmful, how many were correct?
# Recall: Of all the posts that *were* Harmful, how many did we find?
print(classification_report(y_test, y_pred, target_names=['Not Harmful (0)', 'Harmful (1)']))


### --- 8. TEST THE MODEL WITH YOUR OWN SENTENCES --- ###
print("\n--- LIVE PREDICTION TEST ---")
print("Enter a sentence to see if the model classifies it as Harmful or Not Harmful.")
print("Type 'quit' to exit.")

while True:
    # Get input from the user
    new_sentence = input("\nEnter your sentence: ")
    
    if new_sentence.lower() == 'quit':
        break
    
    # Clean the new sentence
    cleaned_sentence = clean_text(new_sentence)
    
    # Vectorize the sentence using the *same* vectorizer
    sentence_vec = vectorizer.transform([cleaned_sentence])
    
    # Make the prediction
    prediction = model.predict(sentence_vec)
    prediction_proba = model.predict_proba(sentence_vec)

    # Get the confidence score
    confidence = prediction_proba[0][prediction[0]]
    
    # Print the result
    if prediction[0] == 1:
        print(f"   -> Result: HARMFUL (Confidence: {confidence*100:.1f}%)")
    else:
        print(f"   -> Result: NOT HARMFUL (Confidence: {confidence*100:.1f}%)")

print("\nProject finished. Have a great day!")