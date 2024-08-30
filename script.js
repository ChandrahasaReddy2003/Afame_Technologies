document.getElementById('prediction-form').addEventListener('submit', async function(event) {
  event.preventDefault();

  const amt = document.getElementById('amt').value;
  // Collect more input values as needed

  const response = await fetch('/predict', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify({
          amt: amt,
          // Add more fields as needed
      }),
  });

  const result = await response.json();
  document.getElementById('result').innerText = `Prediction: ${result.prediction}`;
});
