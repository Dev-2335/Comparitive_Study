# Flask Comparative Study

This project is a Flask-based web application that allows users to perform a comparative study of seven popular machine learning algorithms using custom datasets. Users can upload a CSV file, specify the input and output fields, and view the accuracy results of each algorithm.

## Features

- **File Upload**: Upload any CSV file to analyze.
- **Dynamic Input/Output Field Selection**: Specify the columns to be used as input and output.
- **Model Comparison**: Evaluate the performance of the following algorithms:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
- **Results Visualization**: Display accuracy scores for each algorithm.

## File Structure

```
flask_comparative_study/
│
├── templates/
│   ├── index.html              # Page to upload CSV file
│   ├── input_output.html       # Page to specify input and output fields
│   ├── results.html            # Page to display model comparison results
│
├── uploads/                    # Directory to store uploaded CSV files
│   └── (uploaded files will appear here)
│
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flask_comparative_study.git
   cd flask_comparative_study
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Follow these steps:
   - Upload your CSV file.
   - Specify the input and output fields.
   - View the accuracy results of the machine learning models.

## Dependencies

The project uses the following Python libraries:

- Flask
- pandas
- numpy
- scikit-learn

Install them using the provided `requirements.txt` file.

## Contributing

Contributions are welcome! If you'd like to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push them to your branch.
4. Submit a pull request.

---

Happy coding!

