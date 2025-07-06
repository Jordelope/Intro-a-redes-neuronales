import math

"""
value.py
--------

Este módulo define la clase Value, una implementación numérica personalizada para el cálculo automático de derivadas (autodiff) y propagación hacia atrás (backpropagation), orientada a redes neuronales y operaciones matemáticas básicas.

La clase Value permite construir grafos computacionales de operaciones matemáticas, almacenar los gradientes asociados a cada nodo y realizar el cálculo automático de derivadas mediante el método backward().

Cada instancia de Value representa un escalar con información adicional sobre su gradiente, los nodos predecesores y la operación que lo generó.
"""

class Value:
    """
    Clase que representa un valor escalar en un grafo computacional para autodiff.

    Atributos:
        data (float): Valor numérico almacenado.
        grad (float): Gradiente del nodo respecto al resultado final.
        _prev (set): Conjunto de nodos Value predecesores (inputs de la operación).
        _op (str): Operación que generó este nodo (ej: '+', '*', 'tanh', etc).
        label (str): Etiqueta opcional para identificar el nodo.
        _backward (func): Función que implementa la propagación del gradiente hacia los predecesores.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        """
        Inicializa un nuevo nodo Value.

        Args:
            data (float): Valor numérico.
            _children (tuple): Nodos Value de los que depende este nodo.
            _op (str): Operación que generó este nodo.
            label (str): Etiqueta opcional.
        """
        self.data = data
        self._prev = set(_children)  # Nodos predecesores en el grafo
        self._op = _op               # Operación que generó el nodo
        self.label = label           # Etiqueta opcional
        self.grad = 0.0              # Gradiente (inicialmente cero)
        self._backward = lambda: None  # Función de backpropagation (por defecto no hace nada)

    def __repr__(self):
        """Representación legible del objeto Value."""
        return f"Value: {self.data}"

    def __add__(self, other):
        """
        Suma dos objetos Value o un Value y un escalar.
        Permite construir el grafo de operaciones y define la propagación del gradiente.
        """
        other = other if isinstance(other, Value) else Value(other)
        out_data = self.data + other.data
        out = Value(out_data, (self, other), '+')

        def _backward():
            # La derivada de la suma respecto a cada operando es 1
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward

        return out

    def __radd__(self, other):
        """Permite la suma con el operando izquierdo escalar."""
        return self + other

    def __mul__(self, other):
        """
        Multiplica dos objetos Value o un Value y un escalar.
        Construye el grafo y define la propagación del gradiente.
        """
        other = other if isinstance(other, Value) else Value(other)
        out_data = self.data * other.data
        out = Value(out_data, (self, other), '*')

        def _backward():
            # Derivadas parciales de la multiplicación
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward

        return out

    def __rmul__(self, other):
        """Permite la multiplicación con el operando izquierdo escalar."""
        return self * other

    def __neg__(self):
        """Negación unaria (-self)."""
        return self * -1

    def __sub__(self, other):
        """Resta: self - other."""
        return self + (-other)

    def __pow__(self, other):
        """
        Potencia: self ** other (solo soporta potencias escalares).
        """
        assert isinstance(other, (int, float)), "only supports int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # Derivada de x^k respecto a x es k*x^(k-1)
            self.grad += out.grad * (other * self.data ** (other - 1))
        out._backward = _backward

        return out

    def __truediv__(self, other):
        """División: self / other."""
        return self * other ** (-1)

    def tanh(self):
        """
        Aplica la función tangente hiperbólica al valor.
        """
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)  # t = (e^(2x) -1) / (e^(2x) +1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # Derivada de tanh(x) respecto a x es 1 - tanh(x)^2
            self.grad += out.grad * (1 - t ** 2)
        out._backward = _backward

        return out

    def exp(self):
        """
        Aplica la función exponencial al valor (e^x).
        """
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            # Derivada de exp(x) respecto a x es exp(x)
            self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def backward(self):
        """
        Realiza la propagación hacia atrás (backpropagation) para calcular los gradientes
        del resultado final respecto a todos los nodos del grafo.
        """
        topo = []
        visited = set()

        def build_topo(v):
            # Construye el orden topológico de los nodos del grafo
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        # El nodo final (self) estará al final de la lista topo

        self.grad = 1.0  # El gradiente del resultado respecto a sí mismo es 1
        for node in reversed(topo):
            # Propaga el gradiente hacia los nodos predecesores
            node._backward()

