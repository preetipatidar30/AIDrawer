import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import subprocess
import os
print(os.getcwd())


# Load data from CSV file
df = pd.read_csv('model\Aialgo.csv')

# Clean the data: replace NaN values with empty strings
data = df.fillna('')

# Replace DOMAIN column values for consistency
data.loc[data['DOMAIN'] == 'INFO', 'DOMAIN'] = 'Information'
data.loc[data['DOMAIN'] == 'MATHS', 'DOMAIN'] = 'Mathematics'
data.loc[data['DOMAIN'] == 'IMAGE', 'DOMAIN'] = 'Imagery'

# Separate features (X) and target variable (Y)
X = data['TITLE']
Y = data['DOMAIN']

# Split the data into training and test sets (80-20 split)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Convert string labels to integer labels using a dictionary
label_to_int = {label: idx for idx, label in enumerate(Y_train.unique())}
int_to_label = {idx: label for label, idx in label_to_int.items()}  # Reverse dictionary

Y_train = Y_train.map(label_to_int)
Y_test = Y_test.map(label_to_int)

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# Fit the vectorizer to X_train and transform X_train and X_test
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Initialize logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_features, Y_train)

# Evaluate the model on the training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data:', accuracy_on_training_data)

# Evaluate the model on the test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data:', accuracy_on_test_data)


# Function to classify a new prompt
def classify_prompt(prompt):
    input_data_features = vectorizer.transform([prompt])
    prediction = model.predict(input_data_features)
    domain_prediction = int_to_label[prediction[0]]  # Convert integer prediction to string
    return domain_prediction


# Dictionary mapping domain names to their associated Python filenames
domain_to_filename = {
    'Mathematics': 'model\mathematics.py',
    'Information': 'model\gemini.py',
    'Imagery': 'model\imagery.py',
    # Add other domain-to-file mappings as needed
}


def run_python_file(filename, input_text):
    try:
        # Open the specified Python file and run it
        result = subprocess.run(["python", filename], input=input_text, capture_output=True, text=True, timeout=25)

        # Print the output of the file in the console
        print("Output:")
        print(result.stdout)
        print("Errors:")
        print(result.stderr)
    except FileNotFoundError:
        print("File not found.")
    except subprocess.TimeoutExpired:
        print("Process execution timed out.")


if __name__ == "__main__":
    while True:
        # Get the user's prompt
        user_prompt = input("Enter a text prompt (type 'stop' to quit): ")

        # Check if the user wants to stop
        if user_prompt.lower() == 'stop':
            break

        # Classify the user's prompt
        domain = classify_prompt(user_prompt)

        # Print the classification result
        print(f'Domain: {domain}')

        # Check if the domain has a corresponding Python file to execute
        if domain in domain_to_filename:
            filename = domain_to_filename[domain]
            print(f'Found domain: {domain}. Executing {filename}')
            run_python_file(filename, user_prompt)
        else:
            print("Domain not recognized.")