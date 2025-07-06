import random
from value import Value

"""
MLP_value.py
------------

Este módulo implementa una red neuronal multicapa (MLP, Multi-Layer Perceptron) sencilla, utilizando la clase Value para el cálculo automático de derivadas y la propagación de gradientes.

Contiene las clases Neuron, Layer y MLP, que modelan respectivamente una neurona individual, una capa de neuronas y la red completa. Cada clase está diseñada para ser fácilmente entrenable mediante backpropagation gracias a la integración con Value.
"""

class Neuron:
    """
    Representa una neurona artificial con activación tanh.

    Cada neurona tiene un vector de pesos y un sesgo, ambos inicializados aleatoriamente.
    La salida de la neurona es: tanh(w·x + b), donde w son los pesos, x la entrada y b el sesgo.
    """
    def __init__(self, nin):
        """
        Args:
            nin (int): Número de entradas (inputs) de la neurona.
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]  # Pesos inicializados aleatoriamente
        self.b = Value(random.uniform(-1, 1))  # Sesgo inicial aleatorio

    def __call__(self, x):
        """
        Calcula la salida de la neurona para una entrada x.
        Args:
            x (list of Value): Entradas a la neurona.
        Returns:
            Value: Salida tras aplicar la función de activación tanh.
        """
        # Producto escalar w·x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # Aplicamos función de activación tanh
        out = act.tanh()
        out.label = 'o'  # Etiqueta opcional para identificar la salida
        return out

    def parameters(self):
        """
        Devuelve todos los parámetros entrenables de la neurona (pesos y sesgo).
        """
        return self.w + [self.b]


class Layer:
    """
    Representa una capa de neuronas.

    Una capa contiene varias neuronas, todas con el mismo número de entradas.
    """
    def __init__(self, nin, nout):
        """
        Args:
            nin (int): Número de entradas a la capa.
            nout (int): Número de neuronas (salidas) de la capa.
        """
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        """
        Calcula la salida de la capa para una entrada x.
        Args:
            x (list of Value): Entrada a la capa.
        Returns:
            list[Value] o Value: Salidas de cada neurona. Si la capa tiene una sola neurona, devuelve un único Value.
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        """
        Devuelve todos los parámetros entrenables de la capa (todos los pesos y sesgos de sus neuronas).
        """
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """
    Multi-Layer Perceptron (MLP): red neuronal compuesta por varias capas densas.

    Permite definir redes de cualquier profundidad y tamaño de capa.
    Matemáticamente, la red implementa una composición de funciones:
        f(x) = L_n(...L_2(L_1(x)))
    donde cada L_i es una capa (Layer) y cada capa aplica una transformación lineal seguida de tanh.
    """
    def __init__(self, nin, nouts):
        """
        Args:
            nin (int): Número de entradas a la red.
            nouts (list of int): Lista con el número de neuronas de cada capa (definiendo la arquitectura).
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        """
        Propaga la entrada x a través de todas las capas de la red.
        Args:
            x (list of Value): Entrada a la red.
        Returns:
            Value o list[Value]: Salida final de la red.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        Devuelve todos los parámetros entrenables de la red (pesos y sesgos de todas las capas).
        """
        return [p for layer in self.layers for p in layer.parameters()]
    
    

        
    

