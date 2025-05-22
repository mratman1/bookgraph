import requests
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
from sklearn.manifold import TSNE


# ----------- Step 1: Input your books -----------
book_list = [
    {"title": "The Stranger", "author": "Albert Camus"},
    {"title": "Things Fall Apart", "author": "Chinua Achebe"},
    {"title": "The Plague", "author": "Albert Camus"},
    {"title": "No Longer at Ease", "author": "Chinua Achebe"},
    {"title": "The Trial", "author": "Franz Kafka"},
    {"title": "Love in the Time of Cholera", "author": "Gabriel Garcia Marquez"},
    {"title": "Into Thin Air", "author": "Jon Krakauer"},
    {"title": "The Perfect Theory", "author": "Pedro G. Ferreira"},
    {"title": "Blood on Snow", "author": "Jo Nesbo"},
    {"title": "Firmament", "author": "Simon Clark"},
    {"title": "The Gun", "author": "Fuminori Nakamura"},
    {"title": "All Souls", "author": "Javier Marias"},
    {"title": "The Little Book of String Theory", "author": "Steven S. Gubser"},
    {"title": "Symmetry: A Journey Into the Patterns of Nature", "author": "Marcus du Sautoy"},
    {"title": "Out in the Open", "author": "Jesus Carrasco"},
    {"title": "Homo Deus: A Brief History of Tomorrow", "author": "Yuval Noah Harari"},
    {"title": "We Have Always Lived in the Castle", "author": "Shirley Jackson"},
    {"title": "When Einstein Walked with Gödel: Excursions to the Edge of Thought", "author": "Jim Holt"},
    {"title": "Kokoro", "author": "Natsume Soseki"},
    {"title": "Cabin 102", "author": "Sherry Garland"},
    {"title": "The Supernova Era", "author": "Cixin Liu"},
    {"title": "A View from the Bridge", "author": "Arthur Miller"},
    {"title": "The Night Manager", "author": "John le Carré"},
    {"title": "The Thief", "author": "Fuminori Nakamura"},
    {"title": "Pacifism as Pathology", "author": "Ward Churchill"},
    {"title": "Profit over People", "author": "Noam Chomsky"},
    {"title": "Nocturnes: Five Stories of Music and Nightfall", "author": "Kazuo Ishiguro"},
    {"title": "Brief Answers to the Big Questions", "author": "Stephen Hawking"},
    {"title": "In Other Words", "author": "Jhumpa Lahiri"},
    {"title": "Into the Wild", "author": "Jon Krakauer"},
    {"title": "Wild", "author": "Cheryl Strayed"},
    {"title": "Clash of Civilizations over an Elevator in Piazza Vittorio", "author": "Amara Lakhous"},
    {"title": "Neutrino Hunters", "author": "Ray Jayawardhana"},
    {"title": "Hao", "author": "Ye Chun"},
    {"title": "All My Sons", "author": "Arthur Miller"},
    {"title": "The Stranger in the Woods", "author": "Michael Finkel"},
    {"title": "The Struggle for Europe", "author": "William I. Hitchcock"},
    {"title": "The Handmaid’s Tale", "author": "Margaret Atwood"},
    {"title": "The Agony of Eros", "author": "Byung-Chul Han"},
]


# ----------- Step 2: Helper functions -----------

def get_openlibrary_work_key(title, author):
    query = f"{title} {author}".replace(" ", "+")
    url = f"https://openlibrary.org/search.json?q={query}"
    response = requests.get(url)
    if response.ok:
        data = response.json()
        if data["docs"]:
            work_key = data["docs"][0].get("key")
            if work_key and work_key.startswith("/works/"):
                return work_key
    return None

def get_work_description(work_key):
    url = f"https://openlibrary.org{work_key}.json"
    response = requests.get(url)
    if response.ok:
        data = response.json()
        desc = data.get('description', '')
        if isinstance(desc, dict):
            return desc.get('value', '')
        elif isinstance(desc, str):
            return desc
    return ""

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------- Step 3: Load local sentence-transformers model -----------

print("Loading sentence-transformers model locally...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ----------- Step 4: Enrich book data with descriptions & embeddings -----------

print("Fetching book descriptions and computing embeddings...")
books_data = []
for book in book_list:
    work_key = get_openlibrary_work_key(book["title"], book["author"])
    description = ""
    if work_key:
        description = get_work_description(work_key)
    if not description:
        description = f"No description found for {book['title']}."
    
    embedding = model.encode(description)
    
    books_data.append({
        "title": book["title"],
        "author": book["author"],
        "description": description,
        "embedding": embedding
    })
    print(f"Processed: {book['title']} by {book['author']}")
    time.sleep(1)  # be nice to the API

# ----------- Step 5: Build graph with similarity edges -----------

G = nx.Graph()

# Add nodes
for b in books_data:
    G.add_node(b["title"], author=b["author"], description=b["description"])

# Add edges if cosine similarity > threshold
threshold = 0.2
print("Pairwise cosine similarities:")
for i, b1 in enumerate(books_data):
    for b2 in books_data[i+1:]:
        if b1["embedding"] is not None and b2["embedding"] is not None:
            sim = cosine_sim(b1["embedding"], b2["embedding"])
            print(f"{b1['title']} - {b2['title']}: {sim:.3f}")
            if sim > threshold:  # Threshold for similarity
                G.add_edge(b1["title"], b2["title"], weight=sim)

# ----------- Step 6: Interactive Plot with Plotly and t-SNE -----------

# Only use books with valid embeddings
valid_books = [b for b in books_data if b["embedding"] is not None]
embeddings = np.array([b["embedding"] for b in valid_books])

# ----------- 1. Cluster nodes using KMeans -----------
n_clusters = 6  # You can adjust based on how many groups you expect
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# ----------- 2. Project to 2D with t-SNE -----------
print("Projecting embeddings to 2D with t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
positions_2d = tsne.fit_transform(embeddings)

# Map book title to index
title_to_index = {b["title"]: i for i, b in enumerate(valid_books)}

# ----------- 3. Prepare edge coordinates with alpha -----------
edge_traces = []  # list to hold individual edge traces

for u, v, data in G.edges(data=True):
    if u in title_to_index and v in title_to_index:
        i, j = title_to_index[u], title_to_index[v]
        x0, y0 = positions_2d[i]
        x1, y1 = positions_2d[j]

        sim = data['weight']
        alpha = np.clip((sim - 0.2) / 0.8, 0, 1)  # normalize similarity to [0,1]
        color = f'rgba(150,150,150,{alpha:.2f})'  # grayscale with alpha

        edge_traces.append(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(width=1, color=color),
            hoverinfo='none',
            showlegend=False
        ))

# ----------- 4. Node trace with cluster-based colors -----------

# Distinct color palette (expand if needed)
cluster_palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
]
node_colors = [cluster_palette[label % len(cluster_palette)] for label in cluster_labels]

node_trace = go.Scatter(
    x=positions_2d[:, 0],
    y=positions_2d[:, 1],
    mode='markers',
    marker=dict(
        size=12,
        color=node_colors,
        line=dict(width=1, color='black')
    ),
    text=[f"<b>{b['title']}</b><br>{b['author']}" for b in valid_books],
    hoverinfo='text'
)

# ----------- 5. Plot -----------

fig = go.Figure(data=edge_traces + [node_trace])

fig.update_layout(
    title="Book Similarity Map (Colored by Cluster, Edge Transparency = Similarity Strength)",
    title_x=0.5,
    showlegend=False,
    plot_bgcolor='white',
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
)
fig.show()

