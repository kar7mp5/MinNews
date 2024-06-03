import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class GPTDatasetV1(Dataset):
    """Custom Dataset for GPT Model training.

    This dataset tokenizes the input text and creates input-target pairs
    using a sliding window approach with the specified stride and max length.

    Args:
        txt (str): The input text to be tokenized.
        tokenizer: The tokenizer used to encode the text.
        max_length (int): The maximum length of the input sequences.
        stride (int): The stride size for creating overlapping sequences.
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        if len(token_ids) <= max_length:
            # Check data is short.
            input_chunk = token_ids[:-1]
            target_chunk = token_ids[1:]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        else:
            # Use a sliding window to chunk the book into overlapping sequences of max_length
            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i:i + max_length]
                target_chunk = token_ids[i + 1: i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(prompts, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    combined_text = " ".join(prompts)  # Combine list of texts into a single string
    dataset = GPTDatasetV1(combined_text, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism for the Transformer model.

    Args:
        d_in (int): The dimensionality of the input.
        d_out (int): The dimensionality of the output.
        context_length (int): The length of the context for the attention mechanism.
        dropout (float): The dropout rate.
        num_heads (int): The number of attention heads.
        qkv_bias (bool, optional): Whether to include bias in the QKV projections. Defaults to False.
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """Forward pass for the Multi-Head Attention mechanism.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying multi-head attention.
        """
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class LayerNorm(nn.Module):
    """Layer Normalization for the Transformer model.

    Args:
        emb_dim (int): The dimensionality of the embeddings.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """Forward pass for Layer Normalization.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation function."""

    def forward(self, x):
        """Forward pass for GELU activation.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor after applying GELU activation.
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Feed Forward Neural Network for the Transformer model.

    Args:
        cfg (dict): Configuration dictionary containing model parameters.
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """Forward pass for the Feed Forward network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the Feed Forward network.
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Single Transformer Block consisting of Multi-Head Attention and Feed Forward network.

    Args:
        cfg (dict): Configuration dictionary containing model parameters.
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """Forward pass for the Transformer Block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the Transformer Block.
        """
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """GPT Model consisting of multiple Transformer Blocks.

    Args:
        cfg (dict): Configuration dictionary containing model parameters.
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
    
    
    
    

def generate_text(model, idx, max_new_tokens, context_size):
    """
    Generate text using a trained GPT model.

    Args:
        model (nn.Module): The GPT model used for text generation.
        idx (torch.Tensor): The input tensor containing token indices.
        max_new_tokens (int): The maximum number of new tokens to generate.
        context_size (int): The context size for the model's attention mechanism.

    Returns:
        torch.Tensor: The generated sequence of token indices.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx


def get_text(model, device, PROMPT):
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(PROMPT)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)

    print(f"\n{'='*50}\n{' '*22}IN\n{'='*50}")
    print("\nInput text:", PROMPT)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=50,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    # print(f"\n\n{'='*50}\n{' '*22}OUT\n{'='*50}")
    # print("\nOutput:", out)
    # print("Output length:", len(out[0]))
    # print("Output text:", decoded_text)
    
    return decoded_text

def generate_prompt(example):
    """
    Generate prompts and responses from the given example dataset.

    Args:
        example (dict): The dataset containing 'instruction' and 'output' keys.

    Returns:
        list: List of generated prompt strings.
    """
    output_texts = []
    for i in range(len(example['instruction'])):
        prompt = f"### Instruction: {example['instruction'][i]}\n\n### Response: {example['output'][i]}<eos>"
        output_texts.append(prompt)
    return output_texts

def train_model(model, dataloader, optimizer, criterion, num_epochs, device):
    """
    Train a GPT model.

    Args:
        model (nn.Module): The GPT model to be trained.
        dataloader (DataLoader): The DataLoader providing the training data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (nn.Module): The loss function used for training.
        num_epochs (int): The number of epochs to train for.
        device (torch.device): The device (CPU or GPU) to perform training on.

    Returns:
        None
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids, target_ids = batch
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss}')


def train(model, device, train_data):
    prompts = generate_prompt(train_data)
    print(prompts)
    dataloader = create_dataloader_v1(prompts, batch_size=10, max_length=256, stride=128, shuffle=False, drop_last=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 50
    train_model(model, dataloader, optimizer, criterion, num_epochs, device)



def process_output(output_text):
    processed_output = output_text.split('<eos>')[0].strip()
    return processed_output



if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    torch.manual_seed(120)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    example_data = {
        'instruction': [
            "Explain the bubble sort algorithm.",
            "Provide a summary of the binary search algorithm.",
            "Summarize the quicksort algorithm.",
            "Give an overview of the breadth-first search algorithm.",
            "Explain the depth-first search algorithm.",
            "Summarize the merge sort algorithm.",
            "Provide a brief description of the insertion sort algorithm.",
            "Give an overview of the Dijkstra's algorithm for shortest path.",
            "Explain the A* search algorithm.",
            "Summarize the Floyd-Warshall algorithm for all-pairs shortest paths."
        ],
        'output': [
            "Bubble sort is a simple sorting algorithm that repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order.",
            "Binary search is an efficient algorithm for finding the position of a target value within a sorted array.",
            "Quicksort is a fast sorting algorithm that divides an array into smaller sub-arrays based on a chosen pivot element.",
            "Breadth-first search explores all the neighbor nodes at the present depth prior to moving on to the nodes at the next depth level.",
            "Depth-first search explores as far as possible along each branch before backtracking.",
            "Merge sort is a divide and conquer algorithm that divides the input array into two halves, recursively sorts the halves, and then merges them.",
            "Insertion sort is a simple sorting algorithm that builds the final sorted array one item at a time.",
            "Dijkstra's algorithm finds the shortest path between nodes in a graph by iteratively selecting the node with the shortest distance from the source node.",
            "A* search is a graph traversal and path search algorithm that finds the shortest path between a start node and an end node.",
            "The Floyd-Warshall algorithm is used to find the shortest paths between all pairs of vertices in a weighted graph."
        ]
    }
    train(model, device, example_data)

    PROMPT = 'Explain the bubble sort algorithm'
    formatted_prompt = f"### Instruction: {PROMPT}\n\n### Response:"

    output_text = get_text(model, device, formatted_prompt)
    processed_output_text = process_output(output_text)
    print(processed_output_text)
