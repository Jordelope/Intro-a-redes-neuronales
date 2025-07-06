# Proyecto: Red Neuronal Multicapa desde Cero (MLP autodiff)

Este repositorio implementa una red neuronal multicapa (MLP, Multi-Layer Perceptron) desde cero en Python, sin usar librerías de deep learning como PyTorch o TensorFlow. El objetivo es didáctico: entender los fundamentos matemáticos y computacionales del aprendizaje profundo, el cálculo automático de derivadas (autodiff) y el entrenamiento mediante backpropagation.

## Estructura y descripción de los archivos principales

- **value.py**: Implementa la clase `Value`, un escalar que almacena su valor, gradiente y la información necesaria para construir el grafo computacional y realizar backpropagation. Es la base de todo el sistema de autodiff.

- **MLP_value.py**: Define las clases `Neuron`, `Layer` y `MLP` (red neuronal multicapa). Cada neurona y capa utiliza objetos `Value` para que todas las operaciones sean diferenciables y entrenables.

- **main.py**: Script de ejemplo que crea una red, define un pequeño dataset, realiza una predicción inicial, entrena la red mediante descenso de gradiente y muestra la predicción final. Es el punto de entrada recomendado para experimentar.

- **draw_dot.py**: Permite visualizar el grafo computacional de una operación o pérdida usando Graphviz. Útil para entender cómo se propagan los gradientes y cómo se construye el grafo de operaciones.

## Conexión entre los archivos

- `main.py` utiliza `MLP_value.py` para crear y entrenar la red, y ambos dependen de `value.py` para el cálculo automático de derivadas.
- `draw_dot.py` puede usarse desde `main.py` (descomentando las líneas correspondientes) para visualizar el grafo de la red y los gradientes.

## Instalación de dependencias

Se requiere Python 3.7+ y Graphviz:

```bash
pip install -r requirements.txt
```

Para la visualización del grafo, es necesario tener instalado Graphviz en el sistema (no solo el paquete Python). Puedes descargarlo desde https://graphviz.gitlab.io/download/ y asegurarte de que el ejecutable esté en tu PATH.

## Uso

Ejecuta el script principal para entrenar y probar la red:

```bash
python main.py
```

Puedes modificar la arquitectura de la red, los datos de entrada o los hiperparámetros directamente en `main.py` para experimentar y observar el comportamiento del aprendizaje.

---

Este proyecto es util para quienes desean comprender los fundamentos de las redes neuronales y el autodiff desde una perspectiva matemática y computacional, sin depender de frameworks externos.
