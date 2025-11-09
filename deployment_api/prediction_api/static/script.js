document.getElementById("predictBtn").addEventListener("click", async () => {
  const checkboxes = document.querySelectorAll('input[name="models"]:checked');
  const selectedModels = Array.from(checkboxes).map(c => c.value);

  const form = document.getElementById("predictionForm");
  const formData = new FormData(form);
  const data = {};

  // Prikupljanje vrednosti iz forme
  for (let [key, value] of formData.entries()) {
    data[key] = value;
  }

  const payload = {
    models: selectedModels,
    data: data
  };

  const resultBox = document.getElementById("result");
  resultBox.style.display = "block"; 
  resultBox.textContent = "⏳ Obrada podataka...";

  try {
    const response = await fetch("/api/predict/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const result = await response.json();

    if (result.status === "success") {
      // Kreiranje lepog ispisa za svaki model
      let html = `<h3>Rezultati predikcije:</h3>`;
      for (let [model, res] of Object.entries(result.results)) {
        html += `<div class="model-result">
          <strong>${model}:</strong><br>
          Predikcija: ${res.prediction} <br>
          Pouzdanost: ${res.confidence} <br>
          Verovatnoća: ${Object.entries(res.probabilities).map(([cls, prob]) => `${cls}: ${prob}`).join(", ")}
        </div><hr>`;
      }
      resultBox.innerHTML = html;
    } else {
      resultBox.textContent = `⚠️ Greška: ${result.message}`;
    }

  } catch (err) {
    console.error(err);
    resultBox.textContent = "⚠️ Greška prilikom slanja zahteva.";
  }
});
