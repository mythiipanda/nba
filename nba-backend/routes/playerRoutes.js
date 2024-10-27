const express = require('express');
const router = express.Router();
const playerController = require('../controllers/playerController');

router.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', 'http://localhost:3000');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  next();
});

router.get('/', playerController.getAllPlayers);
router.get('/search', playerController.searchPlayers);

module.exports = router;
