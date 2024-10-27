const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bodyParser = require('body-parser');
const dotenv = require('dotenv');
const admin = require('firebase-admin');
const serviceAccount = require('./serviceAccountKey.json');
dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});
// Middleware
app.use(cors({
  origin: process.env.REACT_APP_API_SENDER //set sender when not using vercel
}));
app.use(bodyParser.json());
const authenticate = async (req, res, next) => {
  const idToken = req.headers.authorization?.split('Bearer ')[1];
  if (!idToken) {
    return res.status(401).send('Unauthorized');
  }

  try {
    const decodedToken = await admin.auth().verifyIdToken(idToken);
    req.user = decodedToken;
    next();
  } catch (error) {
    return res.status(401).send('Unauthorized');
  }
};
// Connect to MongoDB Atlas
const MONGODB_URL = process.env.MONGODB_URL;
mongoose.connect(MONGODB_URL, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
  .then(() => console.log('Connected to MongoDB Atlas'))
  .catch(err => console.error('Could not connect to MongoDB Atlas', err));

// Define Blog Post Schema
const blogPostSchema = new mongoose.Schema({
  title: String,
  content: String,
  date: { type: Date, default: Date.now }
});

const BlogPost = mongoose.model('BlogPost', blogPostSchema);

app.get('/api/posts', async (_, res) => {
  try {
    const posts = await BlogPost.find().sort('-date');
    res.json(posts);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

app.post('/api/posts', async (req, res) => {
  const { title, content } = req.body;
  const newPost = new BlogPost({ title, content });
  try {
    const savedPost = await newPost.save();
    res.status(201).json(savedPost);
  } catch (err) {
    console.error('Error saving post:', err);
    res.status(400).json({ message: err.message });
  }
});

// Define Player Schema
const playerSchema = new mongoose.Schema({
  PLAYER_ID: { type: Number, required: true },
  SEASON_ID: { type: String, required: true },
  LEAGUE_ID: { type: String, required: true },
  TEAM_ID: { type: Number, required: true },
  TEAM_ABBREVIATION: { type: String, required: true },
  PLAYER_AGE: { type: Number, required: true },
  GP: { type: Number, required: true },
  GS: { type: Number, required: true },
  MIN: { type: Number, required: true },
  FGM: { type: Number, required: true },
  FGA: { type: Number, required: true },
  FG_PCT: { type: Number, required: true },
  FG3M: { type: Number, required: true },
  FG3A: { type: Number, required: true },
  FG3_PCT: { type: Number, required: true },
  FTM: { type: Number, required: true },
  FTA: { type: Number, required: true },
  FT_PCT: { type: Number, required: true },
  OREB: { type: Number, required: true },
  DREB: { type: Number, required: true },
  REB: { type: Number, required: true },
  AST: { type: Number, required: true },
  STL: { type: Number, required: true },
  BLK: { type: Number, required: true },
  TOV: { type: Number, required: true },
  PF: { type: Number, required: true },
  PTS: { type: Number, required: true },
  player_name: { type: String, required: true }
});

const Player = mongoose.model('player_name', playerSchema, 'active_players_2024-2025');

// Endpoint to get player data
app.get('/api/players', async (req, res) => {
  try {
    const players = await Player.find();
    res.json(players);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

// Player search endpoint
app.get('/api/players/search', async (req, res) => {
  const { name } = req.query;
  try {
    const players = await Player.find({ PLAYER: new RegExp(name, 'i') });
    res.json(players);
  } catch (error) {
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// Player details endpoint
app.get('/api/player-details/:id', async (req, res) => {
  const playerId = req.params.id;
  try {
    const playerDetails = await mongoose.connection.db.collection('players_adv_all').findOne({ PLAYER_ID: parseInt(playerId) });
    res.json(playerDetails);
  } catch (error) {
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
