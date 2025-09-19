<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Iris Classifier</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      max-width: 400 px;
      margin: 40 px auto;
      padding: 20 px;
      border: 1 px solid #ddd;
      border-radius: 8 px;
      background: #f9f9f9;
    }
    label {
      display: block;
      margin-top: 15 px;
    }
    input[type="text"] {
      width: 100%;
      padding: 8 px;
      margin-top: 5 px;
      box-sizing: border-box;
    }
    button {
      margin-top: 20 px;
      padding: 10 px 15 px;
      background-color: #0078D7;
      color: white;
      border: none;
      border-radius: 4 px;
      cursor: pointer;
    }
    button:hover {
      background-color: #005EA2;
    }
    #result {
      margin-top: 25 px;
      padding: 15 px;
      background: #e6f7ff;
      border: 1 px solid #91d5ff;
      border-radius: 6 px;
    }
    #error {
      margin-top: 25 px;
      padding: 15 px;
      background: #fff1f0;
      border: 1 px solid #ffa39e;
      border-radius: 6 px;
      color: #cf1322;
    }
  </style>
</head>
<body>
  <h1>Iris Classifier</h1>

  <form id="irisForm">
    <label for="point">Enter Features (comma-separated):</label>
    <input
      type="text"
      id="point"
      name="point"
      placeholder="e.g. 5.1, 3.5, 1.4, 0.2"
      required
    />

    <button type="submit">Predict</button>
  </form>

  <div id="result" style="display:none;"></div>
  <div id="error" style="display:none;"></div>

  <script>
    const form = document.getElementById("irisForm");
    const resultDiv = document.getElementById("result");
    const errorDiv = document.getElementById("error");

    form.addEventListener("submit", async (e) => {
      e.preventDefault();

      resultDiv.style.display = "none";
      errorDiv.style.display = "none";
      resultDiv.innerHTML = "";
      errorDiv.innerHTML = "";

      const input = form.point.value.trim();
      const values = input.split(",").map(v => parseFloat(v.trim()));

      if (values.length !== 4 || values.some(isNaN)) {
        errorDiv.textContent = "Please enter exactly 4 numeric values separated by commas.";
        errorDiv.style.display = "block";
        return;
      }

      const payload = { point: values };

      try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || "Unknown error");
        }

        const result = await res
