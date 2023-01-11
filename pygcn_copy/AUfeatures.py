
import torch
au_to_index = {'AU1': 0, 'AU2': 1, 'AU4': 2, 'AU10': 3, 'AU12': 4, 'AU14': 5, 'AU15': 6, 'AU17': 7, 'AU25': 8}
# Create an embedding layer with one row for each AU index and a specified number of columns (say, 32)
embedding = torch.nn.Embedding(9, 32)

# Generate the node matrix by looking up the embedding vectors for each AU index
node_matrix = torch.zeros(9, 32)
for au, index in au_to_index.items():
    node_matrix[index, :] = embedding(torch.tensor([index]))





def cooccurrence_counts():
    pass


adjacency_matrix = torch.zeros(9, 9)

# Set the entry at (i, j) to the co-occurrence count between AUs i and j
#adjacency_matrix[0, 1] = cooccurrence_counts['AU0', 'AU1']
#adjacency_matrix[0, 2] = cooccurrence_counts['AU0', 'AU2']

#adjacency_matrix[8, 7] = cooccurrence_counts['AU8', 'AU7']
