document.getElementById("predictBtn").addEventListener("click", async () => {
  const checkboxes = document.querySelectorAll('input[name="models"]:checked');
  const selectedModels = Array.from(checkboxes).map(c => c.value);

  const form = document.getElementById("predictionForm");
  const formData = new FormData(form);
  const data = {};

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
      let html = `<h3>Rezultati predikcije:</h3>`;
      
      for (let [model, res] of Object.entries(result.results)) {
        
        let limeHtml = '';
        console.log(res.lime_explanation)
        if (res.lime_explanation && res.lime_explanation.length > 0) {
          limeHtml += `<h4>Faktori uticaja (LIME):</h4><ul>`;
          
          res.lime_explanation.forEach(factor => {
            const feature = factor[0];
            const weight = factor[1];
            
            // Boja na osnovu uticaja: zeleno za negativan uticaj (smanjuje rizik Dijabetesa), crveno za pozitivan (povećava rizik)
            const color = weight > 0 ? 'red' : 'green';
            const sign = weight > 0 ? '+' : '';

            limeHtml += `
              <li>
                <span style="color: ${color}; font-weight: bold;">${sign}${weight.toFixed(4)}</span>: ${feature}
              </li>`;
          });
          
          limeHtml += `</ul>`;
        }
        html += `<div class="model-result">
          <strong>Model ${model}:</strong><br>
          Predikcija: **${res.prediction}** <br>
          Pouzdanost: ${res.confidence} <br>
          Verovatnoća: ${Object.entries(res.probabilities).map(([cls, prob]) => `${cls}: ${prob}`).join(", ")}
          
          ${limeHtml} 
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