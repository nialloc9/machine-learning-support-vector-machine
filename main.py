import numpy, pandas, pickle
from sklearn import preprocessing, model_selection, svm
from os import path

'''
To get datasets to work with go to archives.ics.uci.edu/ml/datasets.html
For this example we are going to use the Breast Cancer data set
'''


def filter_data_frame(df, filter_list):
    return df[filter_list]


def serialize(uri, obj):
    with open(uri, "wb") as file:
        pickle.dump(obj, file)


def read_pickle(uri):
    pickle_in = open(uri, "rb")
    return pickle.load(pickle_in)


def get_trained_classifier(uri, features_to_train, labels_to_train):
    if path.isfile(uri):
        return read_pickle(uri)
    else:
        clf = svm.SVC()
        clf.fit(features_to_train, labels_to_train)
        serialize(uri, clf)
        return clf


data_frame = pandas.read_csv('breast-cancer-wisconsin.data.txt')

data_frame.replace("?", -99999, inplace=True)

data_frame.drop(['id'], 1, inplace=True)

features = numpy.array(data_frame.drop(['class'], 1))

features = preprocessing.scale(features)

labels = numpy.array(data_frame['class'])

features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.2)

classifier = get_trained_classifier('breast_cancer_classifier_nearest_neighbour_svm.pickle', features_train, labels_train)

accuracy = classifier.score(features_test, labels_test)

print("Accuracy: ", accuracy)

# make sure that the array you are going to test with does not appear in the document already
example_measures = numpy.array([[4, 2, 1, 1, 1, 2, 3, 2,  1], [4, 2, 1, 2, 2, 2, 3, 2,  1]])

# change values to between 1 and -1
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = classifier.predict(example_measures)

print(prediction)