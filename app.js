const express = require('express');
const PredictController = require('./controller/predictController');

const app = express();
app.use(express.json());

// Endpoint untuk melakukan prediksi
app.post('/api/predict', PredictController.predict);

// Mulai server
app.listen(3000, () => {
  console.log('Server berjalan di http://localhost:3000/');
});
