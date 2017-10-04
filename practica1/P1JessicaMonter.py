import numpy as np
from random import randint
from numpy.linalg import inv
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import Lasso


#Equipo
#Jessica Monter Gallardo
#Carlos Ivan Barrientos Lopez

def main():
	X = np.random.randint(0,100,size=100)
	y = np.random.randint(0,1000,size=len(X))

	mc(X,y)
	lasso(X,y,231,[231, 232, 233])



def mc(X,y):
	print "Minimos Cuadrados"
	X = X.reshape(-1,1)
	reg = linear_model.LinearRegression()
	reg.fit (X, y)
	dataPred = reg.predict(X)

	plt.scatter(X, y,  color='black')
	plt.plot(X, dataPred, color='blue', linewidth=3)

	plt.xticks(())
	plt.yticks(())

	plt.title("Minimos Cuadrados")

	plt.show()

	#COEFICIENTES print reg.coef_
	#PREDICCION print dataPred




def lasso(X, y, alpha, models_to_plot={}):
	print "lasso"
	X = X.reshape(-1,1)
	lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5, fit_intercept=True)

	lassoreg.fit(X,y)
	y_pred = lassoreg.predict(X)
    
	if alpha in models_to_plot:
		plt.subplot(models_to_plot[alpha])
		plt.tight_layout()
		plt.plot(X,y_pred)
		plt.plot(X,y,'.')
		plt.title('Lasso: %.3g'%alpha)
    
	rss = sum((y_pred-y)**2)
	ret = [rss]
	ret.extend([lassoreg.intercept_])
	ret.extend(lassoreg.coef_)
	return ret


def logit():
	print "logit"



main()