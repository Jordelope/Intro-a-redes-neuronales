"""
main.py
-------

Este archivo ejecuta una prueba de entrenamiento y predicción con una red neuronal multicapa (MLP) definida en MLP_value.py.
Se define un pequeño dataset de ejemplo, se inicializa la red, se realiza una predicción inicial, se entrena la red mediante descenso de gradiente y finalmente se muestra la predicción final.

El objetivo es ilustrar el flujo completo de forward, backward y actualización de parámetros usando la clase Value para autodiff.
"""

from value import Value
from MLP_value import MLP
from draw_dot import draw_dot  # Para graficar el grafo de la red (opcional)


# -----------------------------
# 1. Definición de la red
# -----------------------------
# Creamos un perceptrón multicapa (MLP) con 3 capas ocultas de 4, 4 y 1 neuronas respectivamente,
# y entradas de tamaño 3.
# Recomendacion: Probar diferentes estructuras de capas y tamaños de entrada y salida
len_inputs = 3
layer_strc = [4, 4, 1]
NN = MLP(len_inputs, layer_strc)


# -----------------------------
# 2. Definición del dataset
# -----------------------------
# xs: lista de vectores de entrada (cada uno de tamaño 3)
# ys: valores objetivo para cada entrada
# Recomendacion: Probar diferentes valores y tamaños de xs e ys (Importante que sean correctas las dimensiones de entrada y salida)
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]


# -----------------------------
# 3. Hiperparámetros de entrenamiento
# -----------------------------
n_steps = 1000         # Número de iteraciones de entrenamiento
step_sz = 0.2         # Tasa de aprendizaje (learning rate)


# -----------------------------
# 4. Entrenamiento y evaluación
# -----------------------------
if __name__ == "__main__":

    print("\nESTA ES LA PRIMERA PRUEBA DE RED NEURONAL\n")
    print(f"Trabajamos con un Perceptrón multicapa de estructura {layer_strc} e inputs de tamaño {len_inputs}.\n")

    # Predicción inicial (antes de entrenar)
    ypred_inic = [NN(x) for x in xs]
    loss_inic = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred_inic))

    print("La predicción inicial de la red es :\n", ypred_inic)
    print(f"La pérdida inicial es {loss_inic.data}.\nEl vector objetivo es {ys}\n")

    # Entrenamiento de la red
    print(f"Iniciamos entrenamiento de {n_steps} pasos.")
    for k in range(n_steps):
        # Forward pass: calculamos la predicción y la pérdida
        ypred = [NN(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        # Backward pass: calculamos gradientes
        for p in NN.parameters():
            p.grad = 0.0  # Es importante resetear los gradientes antes de cada backward
        loss.backward()

        # Actualización de parámetros (descenso del gradiente)
        for p in NN.parameters():
            p.data += -step_sz * p.grad

    print(f"Entrenamiento terminado. La pérdida es ahora: {loss.data}.\n")
    print(f"La predicción final es:\n {ypred}.")

    # Para visualizar el grafo de la red y los gradientes, descomentar las siguientes líneas:
    dot = draw_dot(loss)
    dot.render('Esquema_MLP')



    