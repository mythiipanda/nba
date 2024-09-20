import React from 'react';

const BlogPost = ({ title, content, date }) => {
  return (
    <article className="bg-white shadow overflow-hidden sm:rounded-lg mb-8">
      <div className="px-4 py-5 sm:px-6">
        <h3 className="text-lg leading-6 font-medium text-gray-900">{title}</h3>
        <p className="mt-1 max-w-2xl text-sm text-gray-500">{date}</p>
      </div>
      <div className="border-t border-gray-200 px-4 py-5 sm:p-0">
        <div className="sm:px-6 sm:py-5">
          <p className="text-sm text-gray-500">{content}</p>
        </div>
      </div>
    </article>
  );
};

export default BlogPost;