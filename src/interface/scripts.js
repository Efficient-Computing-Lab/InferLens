let history = [];
let base_url = "";

function addMessage(role, text, meta) {
    const $chat = $("#chat");
    const cls = role === "user" ? "user" : "bot";
    const $msg = $("<div>").addClass("msg").addClass(cls);
    const $bubble = $("<div>").addClass("bubble").text(text);
    $msg.append($bubble);
    if (meta) $bubble.append($("<div>").addClass("meta").text(meta));
    $chat.append($msg);
    $chat.scrollTop($chat[0].scrollHeight);
}

function setStatus(text) { $("#status").text(text || ""); }
function setTopStatus(text) { $("#topStatus").text(text || ""); }

async function sendMessage_old() {
    const message = $("#message").val().trim();
    const file = $("#file")[0].files[0];
    if (!message && !file) return;

    $("#send").prop("disabled", true);

    if (message) {
        addMessage("user", message);
        history.push({ role: "user", content: message });
    }

    const form = new FormData();
    form.append("message", message || "(file upload)");
    form.append("history", JSON.stringify(history));
    if (file) form.append("file", file);

    setStatus("Sending…");

    try {
        const res = await fetch(base_url+"/api/chat", { method: "POST", body: form });
        const data = await res.json();

        if (!res.ok) {
            addMessage("bot", "Error: " + (data.error || "Unknown error"));
        } else {
            addMessage("bot", data.reply || "(empty reply)",
                `model: ${data.model} • ${data.t_ms} ms`);
            history.push({ role: "assistant", content: data.reply || "" });
        }
    } catch (e) {
        addMessage("bot", "Network error: " + e.toString());
    } finally {
        $("#send").prop("disabled", false);
        setStatus("");
        $("#message").val("").focus();
        $("#file").val("");
    }
}

async function sendMessage() {
  const message = $("#message").val().trim();
  const file = $("#file")[0].files[0];

  if (!message && !file) return;
  $("#send").prop("disabled", true);

  const form = new FormData();
  form.append("message", message || "(file upload)");
  form.append("history", JSON.stringify(history));
  if (file) form.append("file", file);

  if (message) {
    addMessage("user", message);
    history.push({ role: "user", content: message });
  } else if (file) {
    history.push({ role: "user", content: "(File upload)" });
  }

  setStatus("Streaming…");

  // Create an empty bot bubble to be filled by streaming
  const $chat = $("#chat");
  const $msg = $("<div>").addClass("msg bot");
  const $bubble = $("<div>").addClass("bubble").text("");
  $msg.append($bubble);
  $chat.append($msg);
  $chat.scrollTop($chat[0].scrollHeight);

  let full = "";

  try {
    const res = await fetch(base_url+"/api/chat_stream", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      $bubble.text("Error: " + (err.error || "Unknown error"));
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by blank line \n\n
      let parts = buffer.split("\n\n");
      buffer = parts.pop(); // remainder

      for (const part of parts) {
        // We expect lines like: data: {...}
        const line = part.split("\n").find(l => l.startsWith("data: "));
        if (!line) continue;

        const jsonStr = line.slice(6);
        let evt;
        try { evt = JSON.parse(jsonStr); } catch { continue; }

        if (evt.type === "delta") {
          full += evt.text;
          $bubble.text(full);
          $chat.scrollTop($chat[0].scrollHeight);
        } else if (evt.type === "done") {
          // finalize
        } else if (evt.type === "error") {
            $bubble.text(`Error: ${evt.error}\n${evt.details || ""}`);
        } else if (evt.type === "file" && evt.file && evt.file.url) {
            // append a download link under the current bubble
            const label = evt.file.format ? `Download ${evt.file.format.toUpperCase()}` : "Download file";
            const a = document.createElement("a");
            a.href = evt.file.url;
            a.target = "_blank";
            a.textContent = label;
            const div = document.createElement("div");
            div.className = "meta";
            div.appendChild(a);
            $bubble.append(div);
        }
      }
    }

    history.push({ role: "assistant", content: full });

  } catch (e) {
    $bubble.text("Network error: " + e.toString());
  } finally {
    $("#send").prop("disabled", false);
    setStatus("");
    $("#message").val("").focus();
    $("#file").val("");
  }
}


$("#send").on("click", sendMessage);
$("#message").on("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") sendMessage();
});

$("#newchat").on("click", () => {
    history = [];
    $("#chat").empty();
    setStatus("New chat started.");
    setTimeout(() => setStatus(""), 1500);
});

fetch(base_url+"/health")
    .then(r => r.json())
    .then(h => {
        setTopStatus(`API OK · ${h.model}`);
        setStatus(`API OK • model=${h.model}`);
        setTimeout(() => setStatus(""), 2500);
    })
    .catch(() => {
        setTopStatus("API offline");
        setStatus("API not reachable yet.");
    });