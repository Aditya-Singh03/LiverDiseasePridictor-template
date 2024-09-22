import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Loading the dataset.
liver_dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv', header = None, 
                 names = ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'sgpt', 'sgot', 'TP', 'ALB', 'A/G', 'Label'])

liver_dataset.head(10)

#Looking at the shape of the dataset.
print(f'The dataset has {liver_dataset.shape[0]} rows and {liver_dataset.shape[1]} columns')

#Looking for null values.
liver_dataset.info()

"""As the dataset does not have any null values, we can proceed further."""

#Taking a look at the statistical features of the dataset.
liver_dataset.describe()

#Separting the categoric and continuous columns.
cont_colms = ['Age', 'TB', 'DB', 'Alkphos', 'sgpt', 'sgot', 'TP', 'ALB', 'A/G']
cat_colms = ['Gender']

#Plotting countplot for categorical data.
sns.countplot(x = 'Gender', data = liver_dataset)
plt.xticks(ticks = [0, 1], labels = ['Female', 'Male'])
plt.show()

"""It is always a good idea to visualize categoric data in countplots."""

#Making a function for user customized lineplots.
def line_plot(style,width,height,x_val,y_val,color):
  plt.style.use(style)
  plt.figure(figsize=(width,height))
  plt.title(f"{y_val.name} of the people in the dataset")
  plt.plot(x_val,y_val,color=color)
  plt.show()

#Making lineplots for each column of the dataset to look at the distribution.
for i in cont_colms:
    line_plot('dark_background', 16, 4, [i for i in range(liver_dataset.shape[0])], liver_dataset[i], 'purple')

"""The lineplots above will give us a rough idea about how the dataset is distributed."""

for i in cont_colms:
    sns.histplot(x = i, data = liver_dataset, color = 'red')
    plt.show()

#Making a pairplot (for continuous columns) to find corelations (if any).
columns = cont_colms + ['Label']
sns.pairplot(liver_dataset[columns], hue='Label', palette = ["#8000ff","#da8829"])

#From the histograms plotted earlier, we can see that some columns like Alkphos and sgpt are highly fluctuated.
# Therefore it seems important to normalize the data.
from sklearn.preprocessing import StandardScaler
normalizer = StandardScaler()
liver_dataset[cont_colms] = normalizer.fit_transform(liver_dataset[cont_colms])

e_list = []
for i in liver_dataset['Gender']:
    if i == 'Female':
        e_list.append(0.0)
    elif i == 'Male':
        e_list.append(1.0)

e_list[:10]

liver_dataset.insert(1, 'Sex', e_list)

#Extracting the feature values from the dataset.
liver_dataset = liver_dataset.drop(['Gender'], axis=1)
features = liver_dataset.iloc[:, :10].values

#Extracting the labels from the dataset.
labels = liver_dataset['Label'].values

#Looking if everything has been extracted correctly.
print(features[:10])
print(labels[:10])
liver_dataset.info()

#Splitting the dataset into training dataset and testing dataset.
from sklearn.model_selection import train_test_split
(training_features,
 testing_features,
 training_labels,
 testing_labels) = train_test_split(features, labels, test_size=0.20, random_state=0)

#Checking if the splitting has been done correctly.
print(training_features.shape, testing_features.shape)
print(training_labels.shape, testing_labels.shape)
print(training_features[:2])

#Using Smote.
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 42, sampling_strategy=1)
np.all(np.isfinite(training_features))
training_features[~np.isfinite(training_features)] = 0
testing_features[~np.isfinite(testing_features)] = 0
new_training_features, new_training_labels = sm.fit_resample(training_features, training_labels)
print(new_training_labels.shape)
print(sum(new_training_labels == 2))

#Now we are going to implement the prediction model.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc, roc_auc_score, confusion_matrix)

rfc_classifier = RandomForestClassifier()

#Feeding the model with features and labels.
rfc_classifier.fit(new_training_features, new_training_labels)
rfc_predicted = rfc_classifier.predict(testing_features)

print("Accuracy is ", rfc_classifier.score(testing_features, testing_labels))
rfc_predicted

#Accuracy alone is not a good parameter to judge a prediction model.
#Thus, we are using f1-score, precision and recall to see the performace of the model.
f1 = f1_score(testing_labels, rfc_predicted)
print ("F1 Score : \n", f1)
print(roc_auc_score(testing_labels, rfc_predicted))

conf_mat = confusion_matrix(testing_labels, rfc_predicted)
print ("Confusion Matrix : \n", conf_mat)

#Below is the ROC curve for the model.
# ROC curve.
fpr, tpr, thresholds = roc_curve(testing_labels, rfc_predicted, pos_label=2)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for heart stroke prediction')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

import xgboost as xg
from sklearn.metrics import confusion_matrix,classification_report
model = xg.XGBClassifier()
model.fit(new_training_features, new_training_labels),
pred_xg = model.predict(np.array(testing_features))

print(confusion_matrix(testing_labels,pred_xg))
print(classification_report(testing_labels,pred_xg))
print("Accuracy is ", model.score(testing_features, testing_labels))

accuracy_score(training_labels, model.predict(training_features))

