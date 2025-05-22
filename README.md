# bookgraph
A tool for users to find connections between books they have read. It embeds the OpenLibrary descriptions of books inputted into the list as high dimensional vectors using the sentence transformers. It then uses cosine similarity between books to connect them, if this similarity is above a certain treshold. It then also clusters the books using KMeans and colors the clusters. 

The NLP model used is all-MiniLM-L6-v2 from HuggingFace. 