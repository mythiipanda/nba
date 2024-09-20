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

// Define Player Schema
const playerSchema = new mongoose.Schema({
  name: String,
  assists: Number,
  fg_pct: Number,
  free_throw_pct: Number,
  is_active: Boolean,
  points: Number,
  rebounds: Number,
  team: String,
  three_pt_pct: Number,
});

const Player = mongoose.model('Player', playerSchema);

// Endpoint to get player data
app.get('/api/players', async (req, res) => {
  try {
    const players = await Player.find();
    res.json(players);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});