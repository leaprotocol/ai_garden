from sentence_transformers import SentenceTransformer
import torch

# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

# Input text and candidate labels
text = "The movie could be a scifi but it unsurprisingly was not."
candidate_labels = ["action", "comedy", "drama", "sci-fi", "urgent","swearing a lot"]

# Generate embeddings for the text and labels
text_embedding = model.encode(text, convert_to_tensor=True)  # Shape: [embedding_dim]
label_embeddings = model.encode(candidate_labels, convert_to_tensor=True)  # Shape: [num_labels, embedding_dim]

# Normalize the embeddings (to compute cosine similarity)
text_embedding = text_embedding / torch.norm(text_embedding, dim=-1, keepdim=True)
label_embeddings = label_embeddings / torch.norm(label_embeddings, dim=-1, keepdim=True)

# Compute cosine similarity
similarity_scores = torch.matmul(text_embedding, label_embeddings.T)  # Shape: [1, num_labels]
similarity_scores = similarity_scores.squeeze(0)  # Shape: [num_labels]

# Pair labels with their similarity scores
label_score_pairs = list(zip(candidate_labels, similarity_scores))

# Sort by similarity score (descending order)
label_score_pairs.sort(key=lambda x: x[1], reverse=True)

# Display the results
print("Text:", text)
for label, score in label_score_pairs:
    print(f"Label: {label}, Similarity Score: {score:.4f}")