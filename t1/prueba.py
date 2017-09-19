from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model

def main():
	iris = load_iris()
	logClassifier = linear_model.LogisticRegression(C=1,   random_state=111)
	X, y = iris.data[:-1,:], iris.target[:-1]


	logistic = LogisticRegression()
	logistic.fit(X,y)
	print "Predicted class %s, real class %s" % (logistic.predict(iris.data[-1,:]),iris.target[-1])
	print "Probabilities for each class from 0 to 2: %s" % logistic.predict_proba(iris.data[-1,:])




main()