const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bodyParser = require('body-parser');
const dotenv = require('dotenv');
dotenv.config();  // Add this line to use environment variables

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(bodyParser.json());

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
  PLAYER_ID: Number,
  PLAYER: String,
  TEAM_ID: Number,
  TEAM: String,
  GP: Number,
  MIN: Number,
  FGM: Number,
  FGA: Number,
  FG_PCT: Number,
  FG3M: Number,
  FG3A: Number,
  FG3_PCT: Number,
  FTM: Number,
  FTA: Number,
  FT_PCT: Number,
  OREB: Number,
  DREB: Number,
  REB: Number,
  AST: Number,
  STL: Number,
  BLK: Number,
  TOV: Number,
  PTS: Number,
});

const Player = mongoose.model('Player', playerSchema, 'players_adv');

// Endpoint to get player data
app.get('/api/players', async (req, res) => {
  try {
    const players = await Player.find();
    res.json(players);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});
// player search endpoint
app.get('/api/players/search', async (req, res) => {
  const { name } = req.query;
  try {
    const players = await Player.find({ PLAYER: new RegExp(name, 'i') });
    res.json(players);
  } catch (error) {
    res.status(500).json({ error: 'Internal Server Error' });
  }
});
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});