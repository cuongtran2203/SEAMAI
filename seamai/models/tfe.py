import sys
import time

sys.path.append('../')
import torch
from torch import nn
import numpy as np
import  math
from torchinfo import summary as model_summary
from events import Utils
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model, num_heads=16):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # Initialize dimensions
        self.d_model = d_model  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Dimension of each head's key, query, and value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model)  # Query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation

        self.W_o = nn.Linear(d_model, d_model)  # Output 1 transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):

        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        return x.squeeze(1)

    def forward(self, Q, K, V, mask=None):

        Q = Q.unsqueeze(1)
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)

        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))

        return output


class TransformerEncoder_UNIT(nn.Module):
    def __init__(self, split_layer=0, embed_dim=768, num_layers=5, expansion_factor=0.8, num_heads=16,
                 num_classes=1):
        super(TransformerEncoder_UNIT, self).__init__()
        """
        Args:
            num_layers: number of encoder layers
            expansion_factor: factor which determines number of linear layers in feed forward layer
            n_heads: number of heads in multihead attention
            embed_dim: dimension of the embedding
            expansion_factor: factor which determines output dimension of linear layer
            n_heads: number of attention heads
        Returns:
            out: output of the encoder
        """

        self.split_layer = split_layer + 1
        self.num_classes = num_classes

        array = [embed_dim]
        for i in range(1, num_layers+1):
            next_value = round(array[-1] * expansion_factor)
            array.append(next_value)

        self.attention = MultiHeadAttention(d_model=embed_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.5)

        self.feed_forward = nn.ModuleList([nn.Sequential(
            nn.Linear(array[i], array[i]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(array[i], array[i+1]),
            nn.ReLU(),
            nn.Dropout(0.5),
        ) for i in range(num_layers)])

        self.feed_out = nn.Linear(array[-1], embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(0.5)
        self.fc_out = nn.Linear(embed_dim, num_classes)  # Output transformation



    def forward(self, src, mask=None):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block

        """

        key, query, value = src
        attention_out = self.attention(key, query, value, mask=mask)
        attention_residual_out = attention_out + value
        feed_fwd_out = norm1_out = self.dropout1(self.norm1(attention_residual_out))
        for i, ff in enumerate(self.feed_forward):
            feed_fwd_out = ff(feed_fwd_out)
        feed_fwd_residual_out = self.feed_out(feed_fwd_out) + norm1_out
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))
        return self.fc_out(norm2_out)


class TransformerEncoder_CLIENT(nn.Module):
    def __init__(self, split_layer=0, embed_dim=768, num_layers=5, expansion_factor=0.8, num_heads=16, num_classes=1):
        super(TransformerEncoder_CLIENT, self).__init__()
        """
        Args:
            num_layers: number of encoder layers
            expansion_factor: factor which determines number of linear layers in feed forward layer
            n_heads: number of heads in multihead attention
            embed_dim: dimension of the embedding
            expansion_factor: fator ehich determines output dimension of linear layer
            n_heads: number of attention heads
        Returns:
            out: output of the encoder
        """

        self.split_layer = split_layer+1
        self.num_classes = num_classes

        array = [embed_dim]
        for i in range(1, num_layers + 1):
            next_value = round(array[-1] * expansion_factor)
            array.append(next_value)

        self.attention = MultiHeadAttention(d_model=embed_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.5)

        self.feed_forward = nn.ModuleList([nn.Sequential(
            nn.Linear(array[i], array[i]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(array[i], array[i+1]),
            nn.ReLU(),
            nn.Dropout(0.5),
        ) for i in range(self.split_layer)])


    def forward(self, src, mask=None):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block

        """

        key, query, value = src
        attention_out = self.attention(key, query, value, mask=mask)
        attention_residual_out = attention_out + value
        feed_fwd_out = norm1_out = self.dropout1(self.norm1(attention_residual_out))

        for i,ff in enumerate(self.feed_forward):
            feed_fwd_out = ff(feed_fwd_out)

        return [feed_fwd_out,norm1_out]


class TransformerEncoder_SERVER(nn.Module):
    def __init__(self, split_layer=0, embed_dim=768, num_layers=5, expansion_factor=0.8, num_heads=16, num_classes=1):
        super(TransformerEncoder_SERVER, self).__init__()
        """
        Args:
            num_layers: number of encoder layers
            expansion_factor: factor which determines number of linear layers in feed forward layer
            n_heads: number of heads in multihead attention
            embed_dim: dimension of the embedding
            expansion_factor: fator ehich determines output dimension of linear layer
            n_heads: number of attention heads
        Returns:
            out: output of the encoder
        """

        self.split_layer = split_layer+1
        self.num_classes = num_classes
        array = [embed_dim]
        for i in range(1, num_layers + 1):
            next_value = round(array[-1] * expansion_factor)
            array.append(next_value)

        self.feed_forward = nn.ModuleList([nn.Sequential(
            nn.Linear(array[i], array[i]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(array[i], array[i+1]),
            nn.ReLU(),
            nn.Dropout(0.5),
        ) for i in range(self.split_layer, num_layers)])

        self.feed_out = nn.Linear(array[-1], embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(0.5)
        self.fc_out = nn.Linear(embed_dim, num_classes)  # Output transformation




    def forward(self, src, mask=None):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block

        """

        feed_fwd_out,norm1_out = src

        for i,ff in enumerate(self.feed_forward):
            feed_fwd_out = ff(feed_fwd_out)

        feed_fwd_residual_out = self.feed_out(feed_fwd_out) + norm1_out
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))
        return self.fc_out(norm2_out)


if __name__ == "__main__":

    vids = torch.ones([1,768])
    hbco = torch.ones([1,768])
    num_layers = 10

    model = TransformerEncoder_UNIT(embed_dim=768, num_layers=num_layers, num_heads=16)
    model_summary(model)
    for i in range(num_layers):
        split_layer =i
        client = TransformerEncoder_CLIENT(embed_dim=768, num_layers=num_layers, num_heads=16,split_layer=split_layer)
        server = TransformerEncoder_SERVER(embed_dim=768, num_layers=num_layers, num_heads=16,split_layer=split_layer)

        # model_summary(client)
        # model_summary(server)

        t0 =time.time()
        out = client([hbco, hbco, vids])
        print(out[0].shape)
        prediction = server(out)
        t1=time.time()
        print(t1-t0)


    list_flops_backbone = Utils.calculate_flops(list(model.feed_forward))
    list_flops_backbone_2 = np.sum(Utils.calculate_flops(list(model.children()))) - np.sum(list_flops_backbone)
    list_flops_backbone = list(map(lambda x: np.sum([x, list_flops_backbone_2]), list_flops_backbone))
    print(list_flops_backbone)
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits

    print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")
