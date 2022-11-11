from math import sqrt

import torch
import numpy as np


class TokenEmbedding(torch.nn.Module):
    """
    Pytorch module that converts tokens into embeddings.

    Input dimension is: (batch_size, sequence_length)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(
            self,
            embedding_dimension,
            number_of_tokens
    ):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(number_of_tokens, embedding_dimension)

    def forward(self, x):
        return self.embedding_layer(x)


class PositionalEncoding(torch.nn.Module):
    """
    Pytorch module that creates a positional embedding with the same dimensions as the token embeddings.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(self, embedding_dimension, max_sequence_length):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.max_sequence_length = max_sequence_length
        self.positional_encoding = self.create_positional_encoding()

    def create_positional_encoding(self):
        """
        Creates a positional encoding matrix of size (max_sequence_length, embedding_dimension)
        """
        positional_encoding = np.zeros((self.max_sequence_length, self.embedding_dimension))
        for pos in range(self.max_sequence_length):
            for i in range(0, self.embedding_dimension, 2):
                positional_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / self.embedding_dimension)))
                positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / self.embedding_dimension)))
        return torch.from_numpy(positional_encoding).float()

    def forward(self, x):
        """
        Adds the positional encoding to the token embeddings.
        """
        return x + self.positional_encoding[:x.size(0), :]


class SelfAttention(torch.nn.Module):
    """
    Pytorch module for a self attention layer.
    This layer is used in the MultiHeadedSelfAttention module.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, head_dimension)
    """

    def __init__(self, embedding_dimension, head_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        # Create three linear layers, one for the query, one for the key and one for the value.
        self.query_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.key_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.value_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Compute the self attention.
        """

        # x dimensions are: (batch_size, sequence_length, embedding_dimension)
        # query, key, value dimensions are: (batch_size, sequence_length, head_dimension)
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        # Before we can multiply the two matrices we need to take the transpose of the key
        # If this was a 1D array, we would use transpose (0, 1) but it is a 2D array, or batch of inputs.
        # key_transposed dimensions are: (batch_size, head_dimension, sequence_length)
        key_transposed = key.transpose(1, 2)

        # Calculate the dot product
        # bmm is a batch matrix multiplication, meaning we do the same matrix multiplication for a batch of inputs.
        # dot_product_of_query_and_key dimensions are: (batch_size, sequence_length, sequence_length)
        dot_product_of_query_and_key = torch.bmm(query, key_transposed)

        # Divide by the square root of the head dimension.
        # According to the paper, dividing by the square root of the head dimension leads to more stable gradients.
        # dot_product_of_query_and_key_divided dimensions are: (batch_size, sequence_length, sequence_length)
        dot_product_of_query_and_key_divided = dot_product_of_query_and_key / sqrt(self.head_dimension)

        # Softmax makes sure all scores are between 0 and 1 and the sum of scores is 1.
        # attention_scores dimensions are: (batch_size, sequence_length, sequence_length)
        attention_scores = self.softmax(dot_product_of_query_and_key_divided)

        # The attention scores are multiplied by the value
        # Values of tokens with high attention score get highlighted because they are multiplied by a larger number,
        # and tokens with low attention score get drowned out because they are multiplied by a smaller number.
        # Output dimensions are: (batch_size, sequence_length, head_dimension)
        return torch.bmm(attention_scores, value)


class MaskedMultiHeadedSelfAttention(torch.nn.Module):
    """
    Pytorch module for a multi head attention layer.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(self, embedding_dimension, head_dimension, number_of_heads):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        self.number_of_heads = number_of_heads

        # Create the self attention modules
        self.self_attentions = torch.nn.ModuleList(
            [SelfAttention(embedding_dimension, head_dimension) for _ in range(number_of_heads)])

        # Create a linear layer to combine the outputs of the self attention modules
        self.output_layer = torch.nn.Linear(number_of_heads * head_dimension, embedding_dimension)

    def forward(self, x, mask):
        """
        Compute the multi head attention.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        mask dimensions are: (batch_size, sequence_length)
        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """
        # Compute the self attention for each head
        # self_attention_outputs dimensions are:
        # (number_of_heads, batch_size, sequence_length, head_dimension)
        self_attention_outputs = [self_attention(x) for self_attention in self.self_attentions]

        # Concatenate the self attention outputs
        # self_attention_outputs_concatenated dimensions are:
        # (batch_size, sequence_length, number_of_heads * head_dimension)
        concatenated_self_attention_outputs = torch.cat(self_attention_outputs, dim=2)

        # Apply the output layer
        # output dimensions are: (batch_size, sequence_length, embedding_dimension)
        return self.output_layer(concatenated_self_attention_outputs) * mask.unsqueeze(2)


class FeedForward(torch.nn.Module):
    """
    Pytorch module for a feed forward layer.

    A feed forward layer is a fully connected layer with a ReLU activation function in between.
    """

    def __init__(self, embedding_dimension, feed_forward_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.feed_forward_dimension = feed_forward_dimension
        self.linear_1 = torch.nn.Linear(embedding_dimension, feed_forward_dimension)
        self.linear_2 = torch.nn.Linear(feed_forward_dimension, embedding_dimension)

    def forward(self, x):
        """
        Compute the feed forward layer.
        """
        return self.linear_2(torch.relu(self.linear_1(x)))


class EncoderLayer(torch.nn.Module):
    """
    Pytorch module for an encoder layer.

    An encoder layer consists of a multi-headed self attention layer, a feed forward layer and dropout.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(self, embedding_dimension, head_dimension, number_of_heads, feed_forward_dimension, dropout_rate):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        self.number_of_heads = number_of_heads
        self.feed_forward_dimension = feed_forward_dimension
        self.dropout_rate = dropout_rate
        self.multi_head_attention = MaskedMultiHeadedSelfAttention(embedding_dimension, head_dimension, number_of_heads)
        self.feed_forward = FeedForward(embedding_dimension, feed_forward_dimension)
        self.dropout_1 = torch.nn.Dropout(dropout_rate)
        self.dropout_2 = torch.nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        """
        Compute the encoder layer.
        """
        # Apply the multi head attention
        # x dimensions are: (batch_size, sequence_length, embedding_dimension)
        # attention_output dimensions are: (batch_size, sequence_length, embedding_dimension)
        multi_head_attention_output = self.multi_head_attention(x)

        # Apply the dropout
        # Dropout makes sure the model does not overfit the training data by randomly setting some activations to 0.
        multi_head_attention_output = self.dropout_1(multi_head_attention_output)

        # Apply the residual connection
        # The residual connection bypasses the attention layer and adds the input to the output of the attention layer.
        multi_head_attention_output = x + multi_head_attention_output

        # Apply the layer normalization
        # Layer normalization makes sure the output of the attention layer has a mean of 0 and a standard deviation of 1.
        multi_head_attention_output = torch.nn.LayerNorm(multi_head_attention_output.size()[1:]).forward(multi_head_attention_output)

        # Apply the feed forward
        # The feed forward layer is a fully connected layer with a ReLU activation function in between.
        # feed_forward_output dimensions are: (batch_size, sequence_length, embedding_dimension)
        feed_forward_output = self.feed_forward(multi_head_attention_output)

        # Apply the second dropout.
        feed_forward_output = self.dropout_2(feed_forward_output)

        # Apply the residual connection that bypasses the feed forward layer and adds the input to the output of the feed forward layer.
        feed_forward_output = multi_head_attention_output + feed_forward_output

        # Apply the layer normalization
        feed_forward_output = torch.nn.LayerNorm(feed_forward_output.size()[1:]).forward(feed_forward_output)

        # Apply the mask if it is not None
        if mask is not None:
            feed_forward_output = feed_forward_output * mask

        return feed_forward_output


class Encoder(torch.nn.Module):
    """
    Pytorch module for the encoder.

    An Encoder consists of multiple encoder layers stacked on top of each other.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(self, embedding_dimension, head_dimension, number_of_heads, feed_forward_dimension, dropout_rate, number_of_encoder_layers):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        self.number_of_heads = number_of_heads
        self.feed_forward_dimension = feed_forward_dimension
        self.dropout_rate = dropout_rate
        self.number_of_encoder_layers = number_of_encoder_layers
        # Create the encoder layers
        self.encoder_layers = torch.nn.ModuleList(
            [EncoderLayer(embedding_dimension, head_dimension, number_of_heads, feed_forward_dimension, dropout_rate) for _ in range(number_of_encoder_layers)])

    def forward(self, x, mask=None):
        """
        Compute the encoder.
        """
        # Apply the encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return x

class DecoderLayer(torch.nn.Module):
    pass

class Runner(torch.nn.Module):

    def __init__(self):
        pass

    def run(self):
        embedding_dimension = 128
        head_dimension = 64
        max_sequence_length = 12
        number_of_tokens = 27

        # Numpy array containing "cats rule the world" converted into tokens
        # It is a 2 dimensional array, because later on we might want to use batches as input
        input = np.array([[2, 0, 19, 18, 26, 17, 20, 11, 4, 26, 19, 7, 4, 26, 22, 14, 17, 11, 3]])

        # Truncate the input to the first 12 tokens from the sequence, because our
        # max_sequence_length only allows 12 tokens at a time
        truncated_input = input[:, :12]

        # Create the token embedding layer
        token_emb_layer = TokenEmbedding(
            embedding_dimension=embedding_dimension,
            number_of_tokens=number_of_tokens
        )

        # Create the positional encoding layer
        pos_enc_layer = PositionalEncoding(
            embedding_dimension=embedding_dimension,
            max_sequence_length=max_sequence_length
        )

        # Convert the input into a pytorch tensor
        torch_input = torch.from_numpy(truncated_input)
        print(torch_input.shape)  # torch.Size([1, 12])

        # Convert tokens into embeddings
        embeddings = token_emb_layer.forward(torch_input)
        print(embeddings.shape)  # torch.Size([1, 12, 128])

        positional_encoding = pos_enc_layer.forward(embeddings)
        print(positional_encoding.shape)  # torch.Size([1, 12, 128])

        mha = MaskedMultiHeadedSelfAttention(
            embedding_dimension=embedding_dimension,
            head_dimension=embedding_dimension,
            number_of_heads=6
        )

        enc_output = mha.forward(positional_encoding)
        print(enc_output.shape)  # torch.Size([1, 12, 128])

        encoder = Encoder(
            embedding_dimension=embedding_dimension,
            head_dimension=head_dimension,
            number_of_heads=6,
            feed_forward_dimension=embedding_dimension,
            dropout_rate=0.1,
            number_of_encoder_layers=6
        )
        encoder_output = encoder.forward(positional_encoding)
        print(encoder_output.shape)  # torch.Size([1, 12, 128])


Runner().run()
