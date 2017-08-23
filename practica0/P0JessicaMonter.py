import numpy as np
from random import randint
from numpy.linalg import inv
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

#Equipo
#Jessica Monter Gallardo
#Carlos Iván Barrientos López



#Ejercicio 1

#Funcion que permite al usuario introducir los valores de los vectores de la base a ortonormalizar
def entradaVectores():
	print "ORTONORMALIZACION EN R3\n"
	v1 = raw_input("Introduzca las entradas del primer vector separadas por espacios, ej. 1 2 3\n").split()
	v2 = raw_input("Introduzca las entradas del segundo vector separadas por espacios, ej. 1 2 3\n").split()
	v3 = raw_input("Introduzca las entradas del tercer vector separadas por espacios, ej. 1 2 3\n").split()
	v1 = map(float, v1)
	v2 = map(float, v2)
	v3 = map(float, v3)
	matrix = creaMatriz(v1,v2,v3)
	print "Los vectores son: "
	imprimeMatriz(matrix)
	return matrix

#Funcion que crea una matriz partir de 3 vectores
def creaMatriz(v1, v2, v3):
	matrix = np.array([v1, v2, v3])
	return matrix

#Funcion que calcula el producto punto de dos vectores, su division y la multiplicacion del escalar
#resultante por el primer vector
def proyeccion(v1, v2):
    return (np.dot(v1,v2) / np.dot(v1,v1)) * v1


#Funcion que crea la grafica de los vectores contenidos en las matrices.
def graph(mA, mB):

	X, Y, Z, U, V, W = mA[0], mA[1], mA[2], mB[0], mB[1], mB[2]
	fig = plt.figure()
	fig.suptitle('Rojo: Vectores iniciales, Azul: Vectores ortonormales', fontsize=12)
	ax = fig.add_subplot(111, projection='3d')

	ax.quiver(0, 0, 0, X, Y, Z, color='r', pivot='tail')
	ax.quiver(0, 0, 0, U, V, W, color='b', pivot='tail')

	ax.set_xlim([-1, 1])
	ax.set_ylim([-1, 1])
	ax.set_zlim([-1, 1])

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	
	plt.show()



#Funcion que implementa el metodo de Gram-Schmidt para R3
#Devuelve la matriz con los vectores ortonormalizados
def gramSchmidt(mA):
	v1 = mA[0]
	v2 = mA[1]-(proyeccion(v1,mA[1]))
	v3 = mA[2]-(proyeccion(v1,mA[2])-proyeccion(v2,mA[2]))
	normaV1 = np.linalg.norm(mA[0])
	normaV2 = np.linalg.norm(mA[1])
	normaV3 = np.linalg.norm(mA[2])

	matrix = creaMatriz(v1*(1/normaV1),v2*(1/normaV2),v3*(normaV3))
	return matrix

#Funcion que manda a llamar la entrada del usuario y el metodo de Gram-Schmidt    
def primerEjercicio():
	V = entradaVectores()
	if np.linalg.det(V) == 0:
		print "Hay vectores linealmente dependientes."
	else:
		resultado = gramSchmidt(V)
		print("Los vectores ortonormalizados son: \n")
		imprimeMatriz(resultado)
		matrix = creaMatriz(V[0]*(1/np.linalg.norm(V[0])),V[1]*(1/np.linalg.norm(V[1])),V[2]*(1/np.linalg.norm(V[2])))

		graph(matrix,resultado)

	return 0

#Ejercicio 2    

#Funcion que transforma una matriz en cadena para visualizacion en consola
def cadenaMatriz(matriz):
	m = ""
	for vec in matriz:
		for elemento in vec:
				m += "".join(str(elemento))+" "
		m += "\n"
	return m

#Funcion que genera una matriz con numeros aleatorios
def generaMatriz():
	dimension = randint(2,7)
	matrix = np.random.randint(-10,10,size=(dimension,dimension))
	return matrix


#Funcion que imprime una matriz en consola	
def imprimeMatriz(matrix):
	dat = cadenaMatriz(matrix)
	print dat

#Funcion que multiplica matrices revisando si son compatibles para la operacion
def multiplicaMatrices(mA, mB):
	if mA.shape[0] == mB.shape[1]:
		resultado = np.dot(mA, mB)
	else:
	 	print "La cantidad de filas y columnas de las matrices no coinciden"
	 	resultado = np.zeros((4,4))
	return resultado

#Funcion que calcula la inversa de una matriz cuadrada revisando si su determinante es adecuado
def inversaMatrix(mA):
	if np.linalg.det(mA) != 0:
		inversa = np.linalg.inv(mA)
	else:
		print "El determinante de la matriz es cero, no tiene inversa"
		inversa = np.zeros((4,4))
	return inversa

#Funcion que cambia valores de 1 a True, 0 a False en una matriz
def cambioValores(mA):
	mTemp = np.empty([mA.shape[0], mA.shape[0]], dtype=bool )
	for i in range(0,mA.shape[0]):
		for j in range(0,mA.shape[0]):
			if mA[i][j] == 1:
				mTemp[i][j] = True
			else:
				mTemp[i][j] = False
	return mTemp

#Funcion que selecciona valoress de las matrices A o B y los copia a la matriz C segun el criterio
def seleccionValores(mA, mB, mC):
	mTemp = np.empty([mA.shape[0], mA.shape[0]], dtype=int )
	for i in range(0,mA.shape[0]):
		for j in range(0,mA.shape[0]):
			if mC[i][j] == True:
				mTemp[i][j] = mA[i][j]
			else:
				mTemp[i][j] = mB[i][j]
	return mTemp

#Funcion que cambia a 0 los valores menores a -5 en la matriz mA
def seleccionMenores(mA, valor):
	mTemp = np.empty([mA.shape[0], mA.shape[0]], dtype=int )
	for i in range(0,mA.shape[0]):
		for j in range(0,mA.shape[0]):
			if mA[i][j] < valor:
				mTemp[i][j] = 0
			else:
				mTemp[i][j] = mA[i][j]
	return mTemp

#Funcion que manda a llamar las funciones de operaciones con matrices
def segundoEjercicio():
	print "OPERACIONES CON MATRICES\n"
	dimension = randint(2,7)
	mA = np.random.randint(-10,11,size=(dimension,dimension))
	mB = np.random.randint(-10,11,size=(dimension,dimension))


	print "Matriz A"
	imprimeMatriz(mA)
	print "Matriz B"
	imprimeMatriz(mB)

	mC = multiplicaMatrices(mA, mB)

	print "Matriz C = AB"
	imprimeMatriz(mC)

	mI = inversaMatrix(mA)
	print "Matriz inversa de A"
	imprimeMatriz(mI)

	mM = multiplicaMatrices(mA,mI)
	print "Multiplicacion de la matriz A por su inversa"
	imprimeMatriz(mM)

	mCU = np.random.randint(0,2,size=(dimension,dimension))
	print "Matriz de ceros y unos C"
	imprimeMatriz(mCU)

	mCU = cambioValores(mCU)
	print "Matriz de cambio de valores"
	imprimeMatriz(mCU)

	mSeleccion = seleccionValores(mA, mB, mCU)
	print "Matriz que selecciona valores de las matrices A o B"
	imprimeMatriz(mSeleccion)

	mSeleccionMenores = seleccionMenores(mSeleccion, -5)
	print "Matriz que selecciona valores menores a -5 de la matriz anterior"
	imprimeMatriz(mSeleccionMenores)



primerEjercicio()
segundoEjercicio()