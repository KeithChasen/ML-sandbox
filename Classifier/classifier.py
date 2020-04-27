from sklearn import tree

# input data
features = [[140, 1], [130, 1], [170, 0], [150, 0]]
labels = [0, 0, 1, 1]

# train classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# make prediction
print(clf.predict([[150, 0]]))
