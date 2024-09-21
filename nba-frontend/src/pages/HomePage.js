import React, { useState, useEffect } from 'react';
import axios from 'axios';
import BlogPost from '../components/BlogPost';

const HomePage = () => {
  const [blogPosts, setBlogPosts] = useState([]);
  const [newPost, setNewPost] = useState({ title: '', content: '' });

  useEffect(() => {
    fetchPosts();
  }, []);

  const fetchPosts = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/posts');
      setBlogPosts(response.data);
    } catch (error) {
      console.error('Error fetching posts:', error);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewPost(prevPost => ({ ...prevPost, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('Submitting new post:', newPost); // Debugging log
    try {
      const response = await axios.post('http://localhost:5000/api/posts', newPost);
      console.log('Post created successfully:', response.data); // Debugging log
      setNewPost({ title: '', content: '' });
      fetchPosts();
    } catch (error) {
      console.error('Error creating post:', error);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-4">
      <h1 className="text-3xl font-bold text-gray-900 mb-8">NBA Analytics Blog</h1>
      
      {blogPosts.map(post => (
        <BlogPost 
          key={post._id} 
          title={post.title} 
          content={post.content} 
          date={new Date(post.date).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
        />
      ))}

      <div className="mt-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Create a New Blog Post</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="title" className="block text-sm font-medium text-gray-700">Title</label>
            <input 
              type="text" 
              name="title" 
              id="title" 
              value={newPost.title} 
              onChange={handleInputChange} 
              className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
              required
            />
          </div>
          <div>
            <label htmlFor="content" className="block text-sm font-medium text-gray-700">Content</label>
            <textarea 
              name="content" 
              id="content" 
              value={newPost.content} 
              onChange={handleInputChange} 
              rows="4" 
              className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
              required
            />
          </div>
          <div className="flex justify-end">
            <button 
              type="submit" 
              className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Post
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default HomePage;