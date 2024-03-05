import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import base64
from IPython.display import display, HTML

def main():
    # Importar el conjunto de datos CIFAR-10
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    train_images.shape
    test_images.shape

    # Normalizar las imágenes
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Crear el modelo de clasificación de imágenes
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compilar el modelo
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    history = model.fit(train_images, train_labels, epochs=10)

    # Evaluar el modelo
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)

    # Guardar el modelo
    model.save('cifar10_classification_model.h5')

    # Graficar la historia de entrenamiento
    plt.plot(history.history['accuracy'])
    plt.title('Precisión del Modelo')
    plt.ylabel('Exactitud')
    plt.xlabel('Epoca')
    plt.show()

    # Graficar la historia de pérdida
    plt.plot(history.history['loss'])
    plt.title('Pérdida del Modelo')
    plt.ylabel('Pérdida')
    plt.xlabel('Epoca')
    plt.show()

    # Cargar el modelo
    loaded_model = tf.keras.models.load_model('cifar10_classification_model.h5')

    # Ahora, vamos a hacer una predicción con una imagen de ejemplo
    # Seleccionamos una imagen del conjunto de prueba para hacer la predicción
    index_of_image_to_predict = 0
    image_to_predict = test_images[index_of_image_to_predict]

    # Realizamos la predicción
    predictions = loaded_model.predict(image_to_predict.reshape(1, 32, 32, 3))

    # Las predicciones son una matriz de probabilidades para cada clase (10 clases en este caso)
    # Podemos obtener la clase con mayor probabilidad utilizando argmax
    predicted_class = np.argmax(predictions[0])

    # Imprimimos el resultado
    print("Predicción:", predicted_class)
    print("Etiqueta verdadera:", test_labels[index_of_image_to_predict][0])  # Imprimimos la etiqueta verdadera de la imagen seleccionada

    # Convertir la imagen numpy a una cadena de bytes base64
    image_bytes = image_to_predict.astype('uint8').tobytes()

    # Codificar la cadena de bytes base64
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    # Mostrar la imagen en la página web
    display(HTML(f'<img src="data:image/jpeg;base64,{encoded_image}" />'))

# Ejecutar el código principal
main()
