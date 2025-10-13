document.addEventListener("DOMContentLoaded", () => {
  // Enviar con Enter
  document.getElementById("input").addEventListener("keypress", (e) => {
    if (e.key === "Enter") enviar();
  });

  const fileInput = document.getElementById("audio-file");
  if (fileInput) {
    fileInput.addEventListener("change", async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      displayMessage("Adjuntaste un audio. Transcribiendo…", "bot-message");
      await enviarAudio(file);
      e.target.value = ""; // reset input
    });
  }
});

const recordBtn = document.querySelector(".record");
const stopBtn = document.querySelector(".stop");
const indicator = document.getElementById("recording-indicator");

let mediaRecorder; // Se define acá para poder usarlo en ambos eventos
let chunks = []; // Para guardar los datos del audio
let stream; // Para guardar el stream de audio y detenerlo luego

function setRecordingUI(isRecording) {
  if (indicator) indicator.classList.toggle("hidden", !isRecording);
  if (recordBtn) recordBtn.disabled = isRecording;
  if (stopBtn) stopBtn.disabled = !isRecording;
}

if (recordBtn) {
  recordBtn.onclick = () => {
    // Revisar si el navegador soporta getUserMedia
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then((audioStream) => {
          // Guardar el stream y crear el MediaRecorder
          stream = audioStream;
          mediaRecorder = new MediaRecorder(stream);

          mediaRecorder.start();
          setRecordingUI(true);

          // Guardar datos a medida que se graben
          mediaRecorder.ondataavailable = (e) => {
            chunks.push(e.data);
          };

          // Cuando se detenga la grabación se crea el archivo y se envía
          mediaRecorder.onstop = async () => {
            // Volvemos a la lógica original: crear un Blob como WAV
            const audioBlob = new Blob(chunks, { type: "audio/wav" });
            chunks = [];
            await enviarAudio(audioBlob);

            // Para dejar de usar el mic
            if (stream) {
              stream.getTracks().forEach((track) => track.stop());
            }
            setRecordingUI(false);
          };
        })
        .catch((err) => {
          console.error(`getUserMedia error: ${err}`);
          displayMessage("No se pudo acceder al micrófono.", "bot-message");
        });
    } else {
          displayMessage("Tu navegador no soporta grabación de audio.", "bot-message");
    }
  };
}

if (stopBtn) {
  stopBtn.onclick = () => {
    try {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
      } else {
        // Fallback: detener pistas si quedaron activas
        if (stream) {
          stream.getTracks().forEach((t) => t.stop());
        }
        setRecordingUI(false);
      }
    } catch (e) {
      console.error(e);
      setRecordingUI(false);
    }
  };
}

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

async function enviarAudio(audioBlobOrFile) {
  const formData = new FormData();
  // Volvemos a la lógica original: si es Blob (grabación), no forzar filename
  if (audioBlobOrFile instanceof File) {
    formData.append("audio", audioBlobOrFile, audioBlobOrFile.name);
  } else {
    formData.append("audio", audioBlobOrFile);
  }

  try {
    const response = await fetch("/speech-to-text", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    const userMessage = (data.respuesta || "").trim();

    if (!userMessage) {
      displayMessage("No se obtuvo una transcripción del audio.", "bot-message");
      return;
    }

    // Mostrar la transcripción como mensaje del usuario
    displayMessage(userMessage, "user-message");

    // Enviar mensaje igual que en enviar()
    const res = await fetch("/chatbot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mensaje: userMessage }),
    });
    const data2 = await res.json();
    displayMessage(data2.respuesta, "bot-message");
  } catch (error) {
    console.error("Error al enviar el audio:", error);
    displayMessage("Error: No se pudo conectar con el servidor.", "bot-message");
  }
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
