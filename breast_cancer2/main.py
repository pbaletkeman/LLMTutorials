# importing the Python module
import sklearn

# importing the dataset
from sklearn.datasets import load_breast_cancer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

# loading the dataset
data = load_breast_cancer()

# Organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# looking at the data
print("label_names")
print(label_names)
print()
print("labels")
print(labels)
print()
print("feature_names")
print(feature_names)
print()
print("features")
print(features)
print()

# Organizing the data into Sets.


# splitting the data
train, test, train_labels, test_labels = train_test_split(features, labels, test_size = 0.33, random_state = 42)

# Building the Model

# initializing the classifier
gnb = GaussianNB()

# training the classifier
model = gnb.fit(train, train_labels)

# making the predictions
predictions = gnb.predict(test)

# printing the predictions
print("printing the predictions")
print(predictions)

# Evaluating the trained model’s accuracy.
print()
print("evaluating the accuracy")
print(f"{accuracy_score(test_labels, predictions)*100:.4}%")
