// === MSN Bot JavaScript ===

// Variables globales
let mediaRecorder = null;
let chunks = [];
let stream = null;
let isRecording = false;

// Elementos del DOM
const messageInput = document.getElementById("messageInput");
const messagesContainer = document.getElementById("messages");
const sendBtn = document.getElementById("sendBtn");
const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const typingIndicator = document.getElementById("typing-indicator");
const recordingStatus = document.getElementById("recordingStatus");

// === Inicialización ===
document.addEventListener("DOMContentLoaded", function () {
  initializeApp();
  setupEventListeners();
  displayWelcomeMessage();
});

function initializeApp() {
  // Verificar soporte de navegador
  if (!window.fetch) {
    displaySystemMessage("Tu navegador no es compatible con algunas funciones del chat.");
  }

  // Verificar soporte de getUserMedia
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    recordBtn.disabled = true;
    recordBtn.title = "Grabación de voz no soportada en este navegador";
    recordBtn.innerHTML = "🚫";
  }

  // Focus inicial en el input
  messageInput.focus();
}

function setupEventListeners() {
  // Enter para enviar mensaje
  messageInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      e.preventDefault();
      sendMessage();
    }
  });

  // Mostrar indicador de escritura mientras el usuario escribe
  messageInput.addEventListener("input", debounce(handleTyping, 300));

  // Botones de grabación
  recordBtn.addEventListener("click", startRecording);
  stopBtn.addEventListener("click", stopRecording);

  // Botón de enviar
  sendBtn.addEventListener("click", sendMessage);

  // Controles de ventana (efectos visuales)
  setupWindowControls();
}

function setupWindowControls() {
  const minimizeBtn = document.querySelector(".control-btn.minimize");
  const maximizeBtn = document.querySelector(".control-btn.maximize");
  const closeBtn = document.querySelector(".control-btn.close");

  minimizeBtn?.addEventListener("click", () => {
    // Efecto visual de minimizar
    document.querySelector(".msn-window").style.transform = "scale(0.8)";
    setTimeout(() => {
      document.querySelector(".msn-window").style.transform = "scale(1)";
    }, 200);
  });

  maximizeBtn?.addEventListener("click", () => {
    // Efecto visual de maximizar
    const window = document.querySelector(".msn-window");
    window.classList.toggle("maximized");
  });

  closeBtn?.addEventListener("click", () => {
    if (confirm("¿Estás seguro de que quieres cerrar el chat?")) {
      document.body.innerHTML = "<div style='display:flex;justify-content:center;align-items:center;height:100vh;font-family:Tahoma;'>Chat cerrado</div>";
    }
  });
}

// === Manejo de Mensajes ===
function sendMessage() {
  const message = messageInput.value.trim();

  if (!message) {
    messageInput.focus();
    return;
  }

  if (message.length > 1000) {
    displaySystemMessage("El mensaje es demasiado largo (máximo 1000 caracteres).");
    return;
  }

  // Deshabilitar controles durante el envío
  setControlsEnabled(false);

  // Mostrar mensaje del usuario
  displayMessage(message, "user");

  // Limpiar input
  messageInput.value = "";

  // Mostrar indicador de escritura del bot
  showTypingIndicator();

  // Enviar mensaje al servidor
  fetch("/chatbot", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ mensaje: message }),
  })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      hideTypingIndicator();
      displayMessage(data.respuesta, "bot");
    })
    .catch(error => {
      hideTypingIndicator();
      console.error("Error:", error);
      displayMessage("Error: No se pudo conectar con el servidor. Verifica tu conexión.", "system");
    })
    .finally(() => {
      setControlsEnabled(true);
      messageInput.focus();
    });
}

// === Grabación de Audio ===
function startRecording() {
  if (isRecording) return;

  navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      sampleRate: 44100
    }
  })
    .then(audioStream => {
      stream = audioStream;
      mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      chunks = [];
      isRecording = true;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(chunks, { type: "audio/wav" });
        sendAudioMessage(audioBlob);
        stopMediaStream();
      };

      mediaRecorder.start();
      updateRecordingUI(true);

      console.log("Grabación iniciada");
    })
    .catch(error => {
      console.error("Error al acceder al micrófono:", error);
      displaySystemMessage("No se pudo acceder al micrófono. Verifica los permisos.");
    });
}

function stopRecording() {
  if (!isRecording || !mediaRecorder) return;

  mediaRecorder.stop();
  isRecording = false;
  updateRecordingUI(false);

  console.log("Grabación detenida");
}

function stopMediaStream() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
}

function updateRecordingUI(recording) {
  if (recording) {
    recordBtn.style.display = "none";
    stopBtn.style.display = "flex";
    recordingStatus.style.display = "block";
    sendBtn.disabled = true;
    messageInput.disabled = true;
  } else {
    recordBtn.style.display = "flex";
    stopBtn.style.display = "none";
    recordingStatus.style.display = "none";
    sendBtn.disabled = false;
    messageInput.disabled = false;
  }
}

async function sendAudioMessage(audioBlob) {
  try {
    setControlsEnabled(false);
    displaySystemMessage("Procesando audio...");

    const formData = new FormData();
    formData.append("audio", audioBlob, "audio.wav");

    const response = await fetch("/speech-to-text", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    const transcribedText = data.respuesta;

    if (!transcribedText || transcribedText.trim() === "") {
      displaySystemMessage("No se pudo transcribir el audio. Inténtalo de nuevo.");
      return;
    }

    // Mostrar texto transcrito como mensaje del usuario
    displayMessage(`🎤 ${transcribedText}`, "user");

    // Mostrar indicador de escritura del bot
    showTypingIndicator();

    // Enviar mensaje transcrito al chatbot
    const chatResponse = await fetch("/chatbot", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ mensaje: transcribedText }),
    });

    if (!chatResponse.ok) {
      throw new Error(`HTTP error! status: ${chatResponse.status}`);
    }

    const chatData = await chatResponse.json();
    hideTypingIndicator();
    displayMessage(chatData.respuesta, "bot");

  } catch (error) {
    hideTypingIndicator();
    console.error("Error al procesar audio:", error);
    displaySystemMessage("Error al procesar el mensaje de voz. Inténtalo de nuevo.");
  } finally {
    setControlsEnabled(true);
    messageInput.focus();
  }
}

// === Funciones de UI ===
function displayMessage(message, type) {
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message", `${type}-message`);

  const messageBubble = document.createElement("div");
  messageBubble.classList.add("message-bubble");
  messageBubble.textContent = message;

  const timeSpan = document.createElement("div");
  timeSpan.classList.add("message-time");
  timeSpan.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  messageDiv.appendChild(messageBubble);
  messageDiv.appendChild(timeSpan);

  messagesContainer.appendChild(messageDiv);
  scrollToBottom();
}

function displaySystemMessage(message) {
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("system-message");

  const timestamp = document.createElement("span");
  timestamp.classList.add("timestamp");
  timestamp.textContent = `[${new Date().toLocaleTimeString()}] `;

  messageDiv.appendChild(timestamp);
  messageDiv.appendChild(document.createTextNode(message));

  messagesContainer.appendChild(messageDiv);
  scrollToBottom();
}

function displayWelcomeMessage() {
  setTimeout(() => {
    displayMessage("¡Hola! Soy MSN Bot. ¿En qué puedo ayudarte hoy? 😊", "bot");
  }, 1000);
}

function showTypingIndicator() {
  typingIndicator.style.display = "block";
}

function hideTypingIndicator() {
  typingIndicator.style.display = "none";
}

function scrollToBottom() {
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function setControlsEnabled(enabled) {
  sendBtn.disabled = !enabled;
  messageInput.disabled = !enabled;
  recordBtn.disabled = !enabled || isRecording;
}

// === Utilidades ===
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

function handleTyping() {
  // Aquí se podría implementar lógica para mostrar que el usuario está escribiendo
  // Por ahora solo aseguramos que el botón esté habilitado
  if (messageInput.value.trim().length > 0) {
    sendBtn.style.opacity = "1";
  } else {
    sendBtn.style.opacity = "0.7";
  }
}

// === Manejo de Errores Globales ===
window.addEventListener('error', function (e) {
  console.error('Error global:', e.error);
  displaySystemMessage("Ha ocurrido un error inesperado.");
});

// === Compatibilidad ===
if (!Element.prototype.closest) {
  Element.prototype.closest = function (s) {
    var el = this;
    do {
      if (el.matches(s)) return el;
      el = el.parentElement || el.parentNode;
    } while (el !== null && el.nodeType === 1);
    return null;
  };
}
