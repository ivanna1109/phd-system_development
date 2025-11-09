document.getElementById("predictBtn").addEventListener("click", async () => {
  const modelSelect = document.getElementById("modelSelect");
  const selectedModels = Array.from(modelSelect.selectedOptions).map(o => o.value);
  const jsonText = document.getElementById("jsonInput").value;
  const resultBox = document.getElementById("result");

  let data;
  try {
    data = JSON.parse(jsonText);
  } catch (e) {
    resultBox.textContent = "❌ Neispravan JSON format.";
    return;
  }

  const payload = {
    models: selectedModels,
    data: data
  };

  resultBox.textContent = "⏳ Šaljem zahtev...";

  try {
    const response = await fetch("/api/predict/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await response.json();
    resultBox.textContent = JSON.stringify(result, null, 2);
  } catch (err) {
    resultBox.textContent = "⚠️ Greška prilikom slanja zahteva.";
  }
});
