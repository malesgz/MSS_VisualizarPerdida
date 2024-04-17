async function entrenarModelo() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  
    // Compilación del modelo
    model.compile({
      loss: "meanSquaredError",
      optimizer: "sgd",
    });
  
    // Datos de entrenamiento
    const xs = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1]);
    const ys = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10], [9, 1]);
  
    // Configuración de la visualización de la pérdida
    const surface = {
      name: "Pérdida",
      tab: "Entrenamiento",
    };
    const history = [];
  
    // Callback para registrar la pérdida durante el entrenamiento
    const lossCallback = async (epoch, logs) => {
      history.push({ epoch, loss: logs.loss });
      tfvis.show.history(surface, history, ["loss"]);
    };
  
    // Entrenamiento del modelo con el callback de pérdida
    await model.fit(xs, ys, {
      epochs: 500,
      callbacks: { onEpochEnd: lossCallback },
    });
  
    // Actualización del mensaje de información
    document.getElementById("info").innerText =
      "Modelo entrenado. Listo para predecir.";
  
    // Guardar el modelo entrenado para que esté disponible para la predicción
    window.trainedModel = model;
  }
  
  async function predecir() {
    // Obtener el valor de entrada del usuario
    const inputValue = parseInt(document.getElementById("valor").value);
  
    // Verificar si el modelo está entrenado
    if (!window.trainedModel) {
      alert("Por favor, entrena el modelo antes de hacer una predicción.");
      return;
    }
  
    // Predecir el valor de Y para el valor de X ingresado por el usuario
    const model = window.trainedModel;
    const prediction = model.predict(tf.tensor2d([inputValue], [1, 1]));
  
    // Mostrar la predicción en la página
    const outputField = document.getElementById("output_field");
    outputField.innerText = `Predicción: ${prediction.dataSync()[0]}`;
  }