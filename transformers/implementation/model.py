import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        # Embedding layer is kind of a dict just maps a number (token id) to the a vector
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # According to the transformers paper, multiply the output of the embedding layer by
        # sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        seq_len: The maximum length of the sequence that can be input to the transformer
        dropout: To make the model less overfit
        The Dropout layer randomly zeroes some of the elements of the input tensor with probability p,
        which is given as an input while initializing the dropout layer.
        Note that the Dropout layer only applies while training
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a matrix of size (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # 1 means along the second dimension
        # unsqueeze is similar to np.newaxis
        # For numerical stability, we write (1/(10k)^(2i / d_model)) as:
        # exp((-2i / d) * log(10k))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even indices along the embedding dimension
        pe[:, 0::2] = torch.sin(position * div_term)

        # For odd position, we have cos(pos / (10k ^ (2i / d_model))), (basically the even number less than this number)
        pe[:, 1::2] = torch.sin(position * div_term)

        # Add batch dimension at the 0th axis
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # To save the positional encoding when the model will be saved, register it as a buffer
        # register_buffer is typically used to register a buffer that should not to be considered a model parameter.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Shape of x will be (batch_dim, seq_len, d_model)
        # Since we don't want the model to learn the parameters of pe, set its requires_grad to False
        # using the requires_grad_ method
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        # Make learnable parameters alpha and beta
        # alpha for multiplication and beta for addition
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # If x is of shape (a, b, c) then the last dimension is the row
        # The last dimension is the bth row of ath surface
        # So the mean would be of shape (a, b)
        # But keep the last dimension, that is the shape should be (a, b, 1)
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # FeedForward(x) = max(0, x*W1 + b1)*W2 + b2
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # x will be of shape (batch_size, seq_len, d_model)
        # After applying linear layer 1, it will transform to shape (batch_size, seq_len, d_ff)
        # Example: x of shape (4, 4, 2) and W1 of shape (2, 3)
        # Then, broadcast W1 to (4, 2, 3)
        # For every 2D matrix in x, multiply it with W1
        # Therefore result is a matrix of shape (4, 2, 3)
        # After applying linear layer 1, it will transform to shape (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.h = h

        # d_model should be divisible by h
        assert d_model % h == 0, "Embedding dimension must be divisble by the number of heads"

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    """
    - classmethod: This method is bound to the class and not the instance of the class.
    It can modify the class state that would apply across all instances of the class.
    The first parameter of a classmethod is a reference to the class (cls).

    - staticmethod: This type of method doesn't know anything about the class or instance.
    It can't modify the class state or the instance state. It works like a regular function but belongs to the class's namespace.
    It doesn't take any specific first parameter like self or cls.
    """

    @staticmethod
    def attention(query: torch.Tensor,
                  key: torch.Tensor,
                  value: torch.Tensor,
                  mask,
                  dropout: nn.Dropout):
        d_k = query.shape[-1]

        # Key dimension is (b, h, seq_len, d_k)
        # Transpose last 2 dimensions --> (b, h, d_k, seq_len)
        # When multiplied with query, it will transform to (b, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Before applying softmax, apply mask
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        # This means that the sum of the values of the row in the last dimension will be 1
        attention_scores = attention_scores.softmax(dim = -1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Attention scores will be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """
        If we want some words to not interact with other words, we mask them.
        mask is a (probably 2d) boolean tensor.
        Wherever the value of mask is False, it is filled with a very small number
        so that the interaction between the tokens doesn't happen
        """
        query = self.w_q(q) # (b, seq_len, d_model) --> (b, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        # (b, seq_len, d_model) --> (b, seq_len, h, d_k) --> (b, h, seq_len, d_k)

        # First, convert every row of size d_model to a 2-D matrix of size (h, d_k)
        # So, in these new matrices of size (h, d_k), every row will contain the consecutive
        # d_k elements of the embedding vector which will eventually be a part of the different heads

        # Now, we want h matrices of size (seq_len, d_k)
        # For doing that, iterate through all the seq_len 2-D matrices of size (h, d_k)
        # and take the 0th row from each of these matrices and stack them together to form a matrix
        # of size (seq_len, d_k)
        # Similarly, take ith row from each of these 2-D matrices of size (h, d_k)
        # and stack them together to form the ith matrix of size (seq_len, d_k)
        # At the end, we have h matrices of size (seq_len, d_k)

        # This process is similar to transponsing the 1th and 2th dimension of the matrix of shape
        # (b, seq_len, h, d_k) because the ith rows of the matrices of size (h, d_k)
        # are becoming a part of the ith new 2-d matrices.
        # Which means the element at index (b_i, s, i, j) appears at index (b_i, i, s, j)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value,
                                                                     mask, self.dropout)
        
        # (b, h, seq_len, d_k) --> (b, seq_len, h, d_k) --> (b, seq_len, h*d_k)
        # To ensure that the tensor occupies contigous memory, contigous is used
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1 , self.h * self.d_k)

        # (b, seq_len, d_model) --> (b, seq_len, d_model)
        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    """
    Add & Norm layer.
    """
    
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        """
        First, normalize x and then apply the sublayer.
        Different from what paper does, but widely used implementation.
        `sublayer` is a function.
        Add the output with the input x.
        """
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    """
    Encoder has `n` `EncoderBlock` blocks.
    The output of `i`th block is input to `i+1`th block.
    The output of last one is sent to the decoder.
    It contains one `MultiHeadAttentionBlock`, two `Add&Norm` blocks and one `FeedForward` block.
    `self_attention_block` and `feed_forward_block` are passed in the constructor because we don't want to pass the transformer hyperparameters like d_model and number of heads in the constructor.
    """

    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        # Create a list of 2 residual connections
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout)
                                                   for _ in range(2)])
        
    def forward(self, x, src_mask):
        """
        src_mask is a mask applied to the input of the encoder, because we want to hide
        interaction of padding tokens with other tokens
        """
        # All query, key, value are same in encoder
        x = self.residual_connections[0](
                        x,
                        lambda x: self.self_attention_block(x, x, x, src_mask)
                    )
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x
    

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        """
        layers are the `EncoderBlock` objects (probably)
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):

    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList([ResidualConnection(dropout)
                                                   for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        `x` is the input to the decoder
        `src_mask` is the mask applied to the (query @ key.T) in cross attention, where query is the output of decoder's self_attention layer and key is the `encoder_output`.
        `tgt_mask` is the mask applied in the self attention block of decoder
        """
        # All query, key, value are same in the self attention layer of decoder
        x = self.residual_connections[0](
                        x,
                        lambda x: self.self_attention_block(x, x, x, tgt_mask)
            )
        # Cross attention: key and value are encoder's output and query is the output of self-attention layer of decoder
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        )
        # src_mask is passed because we don't want the padding tokens in the encoder output to interact with the decoder output
        
        x = self.residual_connections[2](
            x,
            self.feed_forward_block
        )
        return x
    

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    """
    Project the decoder's output from embedding to vocabulary.
    That is, convert (seq_len, d_model) to (seq, vocab_size) and provide the log softmax probabilities.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()

        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (b, seq_len, d_model) --> (b, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    
class Transformer(nn.Module):

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    """
    During inference, the output of encoder is reused, therefore instead of having a forward method, make encode and decode methods
    """

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 512,
        N: int = 6, # Number of EncoderBlock and DecoderBlock layers
        h: int = 8, # Number of heads
        dropout: float = 0.1,
        d_ff = 2048
) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the EncoderBlock blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the DecoderBlock blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and the decoder
    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)
    # We want to convert d_model to target vocab size
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer