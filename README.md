# Indian Liver Patient Dataset Analysis and Prediction

This project is an analysis and prediction model for the Indian Liver Patient Dataset (ILPD), aimed at determining the likelihood of liver disease in patients. It employs various machine learning techniques to classify patients into healthy and diseased categories.

## Dataset Description

The ILPD is sourced from the UCI Machine Learning Repository. It comprises 583 instances with 11 attributes, including demographic information (age, gender), liver function tests (total bilirubin, direct bilirubin, alkaline phosphatase, SGPT, SGOT), and additional parameters (total proteins, albumin, albumin/globulin ratio). The target variable, 'Label', indicates the presence or absence of liver disease (1 for diseased, 2 for healthy).

## Project Structure

* **Data Loading and Preprocessing:**
    * Loading the dataset from the UCI repository.
    * Handling missing values (though the dataset has none).
    * Basic statistical analysis.
    * Normalizing continuous variables.
    * Encoding categorical variables.
* **Exploratory Data Analysis (EDA):**
    * Visualizations using seaborn and matplotlib:
        * Count plots for categorical data.
        * Line plots for continuous data distribution.
        * Histograms for feature distributions.
        * Pair plots to identify correlations.
* **Data Splitting:**
    * Splitting the dataset into training (80%) and testing (20%) sets.
    * Addressing class imbalance using SMOTE oversampling.
* **Model Implementation and Evaluation:**
    * Three different models are implemented and compared:
        * **Random Forest Classifier:**
            * Provides good accuracy and robustness.
        * **XGBoost Classifier:**
            * Often outperforms other models in various tasks.
        * **Artificial Neural Network (ANN):**
            * Uses TensorFlow/Keras for deep learning.
    * Evaluation metrics include:
        * Accuracy
        * F1 score
        * Precision
        * Recall
        * ROC-AUC score
        * Confusion matrix

## Usage

1. **Clone the Repository:**
```bash
git clone https://github.com/Aditya-Singh03/indian-liver-patient-analysis.git
```
2. **Install Dependencies:**
```bash
pip install pandas seaborn matplotlib numpy scikit-learn imblearn xgboost tensorflow
```
3. **Run the script:**
```bash
python LiverPredictionModel.py
```
This will execute the analysis and prediction process.

## Key Insights and Results

* The dataset is imbalanced, with a higher proportion of healthy patients. SMOTE is used to mitigate this.
* EDA reveals insights into feature distributions and potential correlations.
* All three models show promising results, with XGBoost and Random Forest often performing slightly better than the ANN.
* Further optimization of model hyperparameters and architecture can potentially improve performance.

## Future Improvements

* Explore additional feature engineering techniques.
* Experiment with different classification algorithms.
* Implement cross-validation for more robust model evaluation.
* Deploy the model as a web application for broader access.

## Disclaimer

This project is for educational purposes only. The predictions made by these models should not be used as a substitute for professional medical advice. Consult a qualified healthcare provider for any concerns regarding liver health.
