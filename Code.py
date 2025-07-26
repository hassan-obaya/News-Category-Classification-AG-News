# ===============================
#       Import Dependencies
# ===============================
import pandas as pd
import numpy as np
import nltk
import re
import string
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ===============================
#       Load and Prepare Data
# ===============================

# Load training and testing datasets
train_df = pd.read_csv('/kaggle/input/ag-news-classification-dataset/train.csv')
test_df = pd.read_csv('/kaggle/input/ag-news-classification-dataset/test.csv')

# Combine both datasets into one DataFrame
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# (Optional) Save combined data
# combined_df.to_csv('combined_data.csv')

# Set 'Class Index' as the DataFrame index
df.set_index('Class Index', inplace=True)

# Concatenate 'Title' and 'Description' into a single text field
df['text'] = df['Title'] + " " + df['Description']

# Adjust labels: convert to 0-based index
df['label'] = df.index - 1 

# Preview the processed DataFrame
print(df.head())

# ===============================
#     Download NLTK Resources
# ===============================
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ===============================
#       Text Preprocessing
# ===============================

# Define stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)                   # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)              # Keep only letters
    text = text.lower()                                 # Convert to lowercase
    tokens = nltk.word_tokenize(text)                   # Tokenize text
    tokens = [lemmatizer.lemmatize(w) for w in tokens 
              if w not in stop_words and w not in string.punctuation]  # Remove stopwords and punctuation
    return ' '.join(tokens)

# Apply preprocessing function to text
df['clean_text'] = df['text'].apply(preprocess_text)

# Ensure labels are categorical integers
df['label'] = df['label'].astype('category').cat.codes

# ===============================
#   Feature Extraction (TF-IDF)
# ===============================
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
#     Train ML Classifiers
# ===============================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Choose one classifier
clf = LogisticRegression(max_iter=1000)
# clf = RandomForestClassifier()
# clf = LinearSVC()

# Train and evaluate
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Show performance report
print(classification_report(y_test, y_pred))

# ===============================
#     Word Cloud Visualization
# ===============================
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Generate and display word cloud for each label
for label in df['label'].unique():
    text = ' '.join(df[df['label'] == label]['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for Category {label}")
    plt.show()

# ===============================
#       Neural Network Model
# ===============================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Convert sparse matrix to dense (required by Keras)
X_dense = X.toarray()
y_cat = to_categorical(y)

# Train/test split for neural network
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_dense, y_cat, test_size=0.2, random_state=42)

# Build a simple feedforward neural network
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_nn.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(y_cat.shape[1], activation='softmax')  # Output layer with softmax for multiclass classification
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_nn, y_train_nn, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate on test set
loss, acc = model.evaluate(X_test_nn, y_test_nn)
print(f"Test Accuracy: {acc:.2f}")

# ===============================
#       XGBoost Classifier
# ===============================
import xgboost as xgb

# Instantiate and train XGBoost model
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_clf.fit(X_train, y_train)

# Predict and report results
y_pred_xgb = xgb_clf.predict(X_test)
print(classification_report(y_test, y_pred_xgb))
