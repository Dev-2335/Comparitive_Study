
# Flask Comparative Study

This project is a Flask-based web application that enables users to conduct a comparative study of seven popular machine learning algorithms using custom datasets. Users can upload their own CSV files, select input and output fields, and evaluate the accuracy of different algorithms on the provided data.

## Key Features

- **File Upload**: Upload a CSV file containing your dataset for analysis.
- **Dynamic Field Selection**: Choose the input (features) and output (target) fields from the uploaded dataset.
- **Algorithm Comparison**: Evaluate the performance of seven popular machine learning algorithms:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Support Vector Machine (SVM)
  - Naive Bayes Classifier
  - K-Nearest Neighbors (KNN)
- **Results Visualization**: View the accuracy scores of each model, helping you assess which algorithm performs best on your dataset.

## File Structure

```
flask_comparative_study/
│
├── templates/
│   ├── index.html              # Page to upload CSV file
│   ├── fields.html       # Page to specify input and output fields
│   ├── results.html            # Page to display model comparison results
│
├── uploads/                    # Directory to store uploaded CSV files
│   └── (uploaded files will appear here)
│
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
```


## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/flask_comparative_study.git
   cd flask_comparative_study
   ```

2. **Set Up a Virtual Environment**:
   Create and activate a virtual environment for the project:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the required Python libraries using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Flask Server**:
   Run the Flask application:
   ```bash
   python app.py
   ```

2. **Access the Application**:
   Open your web browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

3. **Steps for Using the Application**:
   - **Upload Your CSV File**: Upload your dataset in CSV format. The app will process it and display the columns.
   - **Select Input and Output Fields**: Choose the columns from your dataset to use as features (input fields) and the target (output field) for prediction.
   - **Model Evaluation**: The app will train and evaluate the following machine learning algorithms on your dataset:
     - Logistic Regression
     - Decision Tree Classifier
     - Random Forest Classifier
     - Gradient Boosting Classifier
     - Support Vector Machine (SVM)
     - Naive Bayes Classifier
     - K-Nearest Neighbors (KNN)
   - **View Results**: The accuracy of each model will be displayed, allowing you to compare their performance.

## Dependencies

This project relies on the following Python libraries:

- **Flask**: A micro web framework used to build the web application.
- **pandas**: For handling and processing datasets in CSV format.
- **numpy**: For numerical operations and array manipulations.
- **scikit-learn**: For implementing machine learning algorithms and evaluating their performance.

These dependencies can be installed by running the following command:
```bash
pip install -r requirements.txt
```

## Contributing

We welcome contributions to improve and extend the project. If you'd like to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push them to your branch.
4. Submit a pull request with a description of your changes.


---

Happy coding, and enjoy exploring machine learning algorithms with Flask!!

