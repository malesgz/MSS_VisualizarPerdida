// Esta función se encarga de definir, compilar y entrenar al modelo que consiste en una capa densa con una unidad de salida, es decir lo convierte en un modelo de regresión lineal simple.
async function entrenarModelo() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  
    // Compilación del modelo
    model.compile({
      // Error cuadrático medio.
      loss: "meanSquaredError",  
      optimizer: "sgd",
    });
  
    // Datos de entrenamiento xs que son los valores de entrada y ys valores de salida del modelo.
    const xs = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1]);
    const ys = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10], [9, 1]);
  
    // Configuración de la visualización de la pérdida.
    const surface = {
      name: "Pérdida",
      tab: "Entrenamiento",
    };
    const history = [];
  
    // Callback para registrar la pérdida durante el entrenamiento
    const lossCallback = async (epoch, logs) => {
      history.push({ epoch, loss: logs.loss });
      // Se muestra la pérdida con el visor.
      tfvis.show.history(surface, history, ["loss"]);
    };
  
    // Entrenamiento de modelo con los datos y el callback de pérdida.
    await model.fit(xs, ys, {
      epochs: 500,
      callbacks: { onEpochEnd: lossCallback },
    });
  
    // Actualización del mensaje de información.
    document.getElementById("info").innerText =
      "Modelo entrenado. Listo para predecir.";
  
    // Guardar el modelo entrenado para que esté disponible para la predicción.
    window.trainedModel = model;
  }
  
  // Con esta función se realizan las predicciones.
  async function predecir() {
    // Obtener el valor de entrada del usuario con el input de entrada.
    const inputValue = parseInt(document.getElementById("valor").value);
  
    // Verificar si el modelo está entrenado, es un mensaje de alarta para que entrene el modelo.
    if (!window.trainedModel) {
      alert("Por favor, entrena el modelo antes de hacer una predicción.");
      return;
    }
  
    // Se obtiene el modelo entrenado.
    const model = window.trainedModel;
    // Predecir el valor de Y para el valor de X ingresado por el usuario
    const prediction = model.predict(tf.tensor2d([inputValue], [1, 1]));
  
    // Mostrar la predicción en la misma página.
    const outputField = document.getElementById("output_field");
    outputField.innerText = `Predicción: ${prediction.dataSync()[0]}`;
  }