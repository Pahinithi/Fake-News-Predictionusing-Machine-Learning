# Import necessary modules
from flask import Flask, request, render_template
import pickle
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

# Create a Flask application instance
app = Flask(__name__, template_folder='.')

# Load the trained model
with open("fake_news_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

# Create an instance of the PorterStemmer class
port_stem = PorterStemmer()

def stemming(content):
    # Remove all non-alphabetic characters and replace them with spaces
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    
    # Convert the text to lowercase
    stemmed_content = stemmed_content.lower()
    
    # Split the text into individual words
    stemmed_content = stemmed_content.split()
    
    # Stem each word and remove stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    
    # Join the words back into a single string
    stemmed_content = ' '.join(stemmed_content)
    
    return stemmed_content  # Return the processed text

# Define a route for the homepage
@app.route("/")
def home():
    return render_template("index.html") # Render the index.html template

# Define a route for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the form
    content = request.form["content"]

    # Preprocess the content
    stemmed_content = stemming(content)  # Apply the stemming function

    # Convert to array for model prediction
    input_data = np.array([stemmed_content])

    # Make a prediction using the model
    prediction = model.predict(input_data)

    # Determine the output message
    if prediction[0] == 0:
        prediction_text = 'The news is Real'
    else:
        prediction_text = 'The news is Fake'

    # Pass the prediction value to the template
    return render_template("index.html", prediction=prediction_text)

# Start the Flask application
if __name__ == "__main__":
    app.run(debug=True) # Run the application in debug mode for development
