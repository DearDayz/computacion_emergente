document.addEventListener("DOMContentLoaded", (event) => {
  // Escucha la tecla 'Enter' en el campo de entrada
  document.getElementById("input").addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      enviar();
    }
  });
});

const record = document.querySelector(".record");
const stop = document.querySelector(".stop");

let mediaRecorder; // Se define acá para poder usarlo en ambos eventos
let chunks = []; // Para guardar los datos del audio
let stream; // Para guardar el stream de audio y detenerlo luego

record.onclick = () => {
  // Revisar si el navegador soporta getUserMedia
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((audioStream) => {
        // Guardar el stream y crear el MediaRecorder
        stream = audioStream;
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.start();
        console.log(mediaRecorder.state);
        console.log("recorder started");
        record.style.background = "red";
        record.style.color = "black";

        // Guardar datos a medida que se graben?
        mediaRecorder.ondataavailable = (e) => {
          chunks.push(e.data);
        };

        // Cuando se detenga la grabación se crea el archivo y se envía
        mediaRecorder.onstop = (e) => {
          const audioBlob = new Blob(chunks, { type: "audio/wav" });
          chunks = [];
          enviarAudio(audioBlob);

          // Para dejar de usar el mic
          if (stream) {
            stream.getTracks().forEach((track) => track.stop());
          }
        };
      })
      .catch((err) => {
        console.error(`The following getUserMedia error occurred: ${err}`);
      });
  } else {
    console.log("getUserMedia not supported on your browser!");
  }
};

stop.onclick = () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    console.log(mediaRecorder.state);
    console.log("recorder stopped");
    record.style.background = "";
    record.style.color = "";
  }
};

function enviar() {
  const inputField = document.getElementById("input");
  const userMessage = inputField.value.trim();

  if (userMessage === "") {
    return; // No enviar si el mensaje está vacío
  }

  // 1. Mostrar el mensaje del usuario en el chat
  displayMessage(userMessage, "user-message");

  // 2. Enviar el mensaje al backend de Flask
  fetch("/chatbot", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    // Nota: Corregí el error tipográfico 'mensjase' a 'mensaje'
    body: JSON.stringify({ mensaje: userMessage }),
  })
    .then((response) => response.json())
    .then((data) => {
      // 3. Mostrar la respuesta del bot
      displayMessage(data.respuesta, "bot-message");
    })
    .catch((error) => {
      console.error("Error al enviar el mensaje:", error);
      displayMessage(
        "Error: No se pudo conectar con el servidor.",
        "bot-message"
      );
    });

  // 4. Limpiar el campo de entrada
  inputField.value = "";
}

async function enviarAudio(audioBlob) {
  const formData = new FormData();
  formData.append("audio", audioBlob);
  
  // Transcribimos el audio
  const response = await fetch("/speech-to-text", {
    method: "POST",
    body: formData,
  })
  .catch((error) => {
      console.error("Error al enviar el mensaje:", error);
      displayMessage(
        "Error: No se pudo conectar con el servidor.",
        "bot-message"
      );
  });
  const data = await response.json();
  const userMessage = data.respuesta;

  if (userMessage === "") {
    return; // No enviar si el mensaje está vacío
  }


  // Mostrar la transcripción como mensaje del usuario
  displayMessage(data.respuesta, "user-message");
  
  // Enviar mensaje igual que en enviar()
  fetch("/chatbot", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ mensaje: userMessage }),
  })
    .then((response) => response.json())
    .then((data) => {
      // 3. Mostrar la respuesta del bot
      displayMessage(data.respuesta, "bot-message");
    })
    .catch((error) => {
      console.error("Error al enviar el mensaje:", error);
      displayMessage(
        "Error: No se pudo conectar con el servidor.",
        "bot-message"
      );
    });
}

function displayMessage(message, className) {
  const messagesDiv = document.getElementById("messages");
  const messageElement = document.createElement("div");
  messageElement.classList.add("message", className);
  messageElement.textContent = message;

  messagesDiv.appendChild(messageElement);

  // Hacer scroll al final para ver el último mensaje
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
