# 1. Receive the input query (text, image, file, etc.) from the user / express server
# 	1.1 For testing purposes, it will be a text query in variable `query`
# 2. Preprocess the input query
# 3. Use input query to search for stuff using FAISS
#   3.3 Vectors are in some directory (might change location later)
# <Done for this file, vectors is passed to another function for generating response>
# 4. Generate response based on the search results, and then return response to server
