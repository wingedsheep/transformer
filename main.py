import random

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
        return x + self.positional_encoding[:x.size(1), :]


class MaskedSelfAttention(torch.nn.Module):
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
        self.query_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.key_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.value_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, mask):
        """
        Compute the self attention.

        x dimension is: (batch_size, sequence_length, embedding_dimension)
        output dimension is: (batch_size, sequence_length, head_dimension)

        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """

        # x dimensions are: (batch_size, sequence_length, embedding_dimension)
        # query, key, value dimensions are: (batch_size, sequence_length, head_dimension)
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        # Calculate the attention weights.
        # attention_weights dimensions are: (batch_size, sequence_length, sequence_length)
        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dimension)

        # Apply the mask to the attention weights. Set the masked values to a large negative value when the mask is 0.
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)

        # Softmax makes sure all scores are between 0 and 1 and the sum of scores is 1.
        # attention_scores dimensions are: (batch_size, sequence_length, sequence_length)
        attention_scores = self.softmax(attention_weights)

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

    def __init__(self, embedding_dimension, number_of_heads):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = embedding_dimension // number_of_heads
        self.number_of_heads = number_of_heads

        # Create the self attention modules
        self.self_attentions = torch.nn.ModuleList(
            [MaskedSelfAttention(embedding_dimension, self.head_dimension) for _ in range(number_of_heads)])

        # Create a linear layer to combine the outputs of the self attention modules
        self.output_layer = torch.nn.Linear(number_of_heads * self.head_dimension, embedding_dimension)

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
        self_attention_outputs = [self_attention(x, mask) for self_attention in self.self_attentions]

        # Concatenate the self attention outputs
        # self_attention_outputs_concatenated dimensions are:
        # (batch_size, sequence_length, number_of_heads * head_dimension)
        concatenated_self_attention_outputs = torch.cat(self_attention_outputs, dim=2)

        # Apply the output layer to the concatenated self attention outputs
        # output dimensions are: (batch_size, sequence_length, embedding_dimension)
        return self.output_layer(concatenated_self_attention_outputs)


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


class GptDecoderLayer(torch.nn.Module):
    """
    Pytorch module for an encoder layer.

    An encoder layer consists of a multi-headed self attention layer, a feed forward layer and dropout.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(
            self,
            embedding_dimension,
            number_of_heads,
            feed_forward_dimension,
            dropout_rate
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.feed_forward_dimension = feed_forward_dimension
        self.dropout_rate = dropout_rate

        # Create the multi-headed self attention layer
        self.multi_headed_self_attention = MaskedMultiHeadedSelfAttention(embedding_dimension, number_of_heads)

        # Create the feed forward layer
        self.feed_forward = FeedForward(embedding_dimension, feed_forward_dimension)

        # Create the dropout layer
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        """
        Compute the encoder layer.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        mask dimensions are: (batch_size, sequence_length)
        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """
        # Compute the multi-headed self attention
        # multi_headed_self_attention_output dimensions are: (batch_size, sequence_length, embedding_dimension)
        multi_headed_self_attention_output = self.multi_headed_self_attention(x, mask)

        # Apply dropout to the multi-headed self attention output
        # multi_headed_self_attention_output_dropout dimensions are: (batch_size, sequence_length, embedding_dimension)
        multi_headed_self_attention_output_dropout = self.dropout(multi_headed_self_attention_output)

        # Add the multi-headed self attention output to the input
        # multi_headed_self_attention_output_dropout_residual dimensions are: (batch_size, sequence_length, embedding_dimension)
        multi_headed_self_attention_output_dropout_residual = multi_headed_self_attention_output_dropout + x

        # Compute the feed forward layer
        # feed_forward_output dimensions are: (batch_size, sequence_length, embedding_dimension)
        feed_forward_output = self.feed_forward(multi_headed_self_attention_output_dropout_residual)

        # Apply dropout to the feed forward output
        # feed_forward_output_dropout dimensions are: (batch_size, sequence_length, embedding_dimension)
        feed_forward_output_dropout = self.dropout(feed_forward_output)

        # Add the feed forward output to the multi-headed self attention output
        # feed_forward_output_dropout_residual dimensions are: (batch_size, sequence_length, embedding_dimension)
        feed_forward_output_dropout_residual = feed_forward_output_dropout + multi_headed_self_attention_output_dropout_residual

        return feed_forward_output_dropout_residual


class GptDecoder(torch.nn.Module):
    """
    Pytorch module for the GPT-2 decoder.

    The decoder consists of a token embedding layer, a positional encoding layer, and a stack of encoder layers.

    Input dimension is: (batch_size, sequence_length)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(
            self,
            embedding_dimension,
            number_of_tokens,
            number_of_layers,
            number_of_heads,
            feed_forward_dimension,
            dropout_rate,
            max_sequence_length
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_tokens = number_of_tokens
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads
        self.feed_forward_dimension = feed_forward_dimension
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length

        # Create the token embedding layer
        self.token_embedding = TokenEmbedding(embedding_dimension, number_of_tokens)

        # Create the positional encoding layer
        self.positional_encoding = PositionalEncoding(embedding_dimension, max_sequence_length)

        # Create the encoder layers
        self.encoder_layers = torch.nn.ModuleList(
            [GptDecoderLayer(embedding_dimension, number_of_heads, feed_forward_dimension, dropout_rate) for _ in
             range(number_of_layers)])

    def forward(self, x, mask):
        """
        Compute the decoder.

        x dimensions are: (batch_size, sequence_length)
        mask dimensions are: (batch_size, sequence_length)
        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """
        # Compute the token embeddings
        # token_embeddings dimensions are: (batch_size, sequence_length, embedding_dimension)
        token_embeddings = self.token_embedding(x)

        # Compute the positional encoding
        # positional_encoding dimensions are: (batch_size, sequence_length, embedding_dimension)
        positional_encoding = self.positional_encoding(token_embeddings)

        # Compute the encoder layers
        # encoder_outputs dimensions are: (batch_size, sequence_length, embedding_dimension)
        encoder_outputs = positional_encoding
        for encoder_layer in self.encoder_layers:
            encoder_outputs = encoder_layer(encoder_outputs, mask)

        return encoder_outputs


class LMHead(torch.nn.Module):
    """
    Pytorch module for the language model head.

    The language model head is a linear layer that is applied to the output of the decoder.
    """

    def __init__(self, embedding_dimension, number_of_tokens):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_tokens = number_of_tokens
        self.linear = torch.nn.Linear(embedding_dimension, number_of_tokens)

    def forward(self, x):
        """
        Compute the language model head.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        """
        # Compute the linear layer
        # linear_output dimensions are: (batch_size, sequence_length, number_of_tokens)
        linear_output = self.linear(x)

        return linear_output


class GPT(torch.nn.Module):
    """
    Pytorch module for an autoregressive GPT model.
    """

    def __init__(
            self,
            embedding_dimension,
            number_of_tokens,
            number_of_layers,
            number_of_heads,
            feed_forward_dimension,
            dropout_rate,
            max_sequence_length
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_tokens = number_of_tokens
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads
        self.feed_forward_dimension = feed_forward_dimension
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length

        # Create the decoder
        self.decoder = GptDecoder(
            embedding_dimension,
            number_of_tokens,
            number_of_layers,
            number_of_heads,
            feed_forward_dimension,
            dropout_rate,
            max_sequence_length
        )

        # Create the language model head
        self.lm_head = LMHead(embedding_dimension, number_of_tokens)

        # Initialize the weights of the network
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the network.
        """
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, mask):
        """
        Compute the GPT model.

        x dimensions are: (batch_size, sequence_length)
        mask dimensions are: (batch_size, sequence_length)
        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """
        # Compute the decoder
        # decoder_outputs dimensions are: (batch_size, sequence_length, embedding_dimension)
        decoder_outputs = self.decoder(x, mask)

        # Compute the language model head
        # lm_head_outputs dimensions are: (batch_size, sequence_length, number_of_tokens)
        lm_head_outputs = self.lm_head(decoder_outputs)

        return lm_head_outputs


class AutoregressiveWrapper(torch.nn.Module):
    """
    Pytorch module that wraps a GPT model and makes it autoregressive.

    The GPT model is autoregressive because the output of the model is used as the input of the model.
    """

    def __init__(self, gpt_model):
        super().__init__()
        self.model = gpt_model

    def forward(self, x, mask):
        """
        Autoregressive forward pass
        """
        inp, target = x[:, :-1], x[:, 1:]
        mask = mask[:, 1:]

        output = self.model(inp, mask)
        return output, target

    def generate(self, start_tokens, seq_len, temperature=1.0, eos_token=None, pad_value=0):
        """
        Generate text with the GPT model.
        """
        self.model.eval()

        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        out = start_tokens

        for _ in range(seq_len):
            x = out[:, -self.model.max_sequence_length:]

            mask = torch.ones_like(x)

            # Mask padding tokens
            mask[x == pad_value] = 0

            logits = self.model(x, mask)[:, -1]

            if temperature != 1.0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            out = torch.cat([out, next_token], dim=1)

            if eos_token is not None and next_token == eos_token:
                break

        return out


class Tokenizer:

    def __init__(self):
        self.dictionary = {
            '<PAD>': 0,
            '<BOS>': 1,
            '<EOS>': 2,
            'a': 3,
            'b': 4,
            'c': 5,
            'd': 6,
            'e': 7,
            'f': 8,
            'g': 9,
            'h': 10,
            'i': 11,
            'j': 12,
            'k': 13,
            'l': 14,
            'm': 15,
            'n': 16,
            'o': 17,
            'p': 18,
            'q': 19,
            'r': 20,
            's': 21,
            't': 22,
            'u': 23,
            'v': 24,
            'w': 25,
            'x': 26,
            'y': 27,
            'z': 28,
            ' ': 29,
            '1': 30,
            '2': 31,
            '3': 32,
            '4': 33,
            '5': 34,
            '6': 35,
            '7': 36,
            '8': 37,
            '9': 38,
            '0': 39
        }
        self.reverse_dictionary = {v: k for k, v in self.dictionary.items()}

    def tokenize(self, text):
        tokens = []
        for character in text:
            if character in self.dictionary:
                tokens.append(self.dictionary[character])
        return tokens

    def character_to_token(self, character):
        return self.dictionary[character]

    def token_to_character(self, token):
        return self.reverse_dictionary[token]


class Runner(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def run(self):
        embedding_dimension = 128
        feed_forward_dimension = 256
        max_sequence_length = 20
        number_of_tokens = 40

        # Create the tokenizer
        tokenizer = Tokenizer()

        # Create the model
        model = AutoregressiveWrapper(GPT(
            embedding_dimension=embedding_dimension,
            number_of_tokens=number_of_tokens,
            number_of_heads=4,
            number_of_layers=3,
            feed_forward_dimension=feed_forward_dimension,
            dropout_rate=0.1,
            max_sequence_length=max_sequence_length
        ))

        # Create the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # Create the loss function
        loss_function = torch.nn.CrossEntropyLoss()

        # Create the training data
        training_data = '. '.join([
            'cats rule the world',
            'dogs are the best',
            'elephants have long trunks',
            'monkeys like bananas',
            'pandas eat bamboo',
            'tigers are dangerous',
            'zebras have stripes',
            'lions are the kings of the savannah',
            'giraffes have long necks',
            'hippos are big and scary',
            'rhinos have horns',
            'penguins live in the arctic',
            'polar bears are white'
        ])

        # Tokenize the training data
        tokenized_training_data = tokenizer.tokenize(training_data)

        for _ in range(max_sequence_length):
            # Prepend padding tokens
            tokenized_training_data.insert(0, tokenizer.character_to_token('<PAD>'))

        # Create sequences of length max_sequence_length + 1
        # The last token of each sequence is the target token
        sequences = []
        for i in range(0, len(tokenized_training_data) - max_sequence_length - 1):
            sequences.append(tokenized_training_data[i: i + max_sequence_length + 1])

        # Train the model
        batch_size = 1
        for epoch in range(200):
            losses = []

            # Shuffle the sequences
            random.shuffle(sequences)

            # Create batches of sequences
            batches = []
            for i in range(0, len(sequences), batch_size):
                batches.append(sequences[i: i + batch_size])

            # Train the model on each batch
            for batch in batches:
                model.train()

                # Create the input and target tensors
                input_tensor = torch.zeros((batch_size, max_sequence_length + 1), dtype=torch.long)

                for i, sequence in enumerate(batch):
                    input_tensor[i] = torch.tensor(sequence)

                # Create the mask tensor
                mask_tensor = torch.ones((batch_size, max_sequence_length + 1), dtype=torch.long)

                # Compute the model output
                model_output, target = model.forward(input_tensor, mask_tensor)

                # Compute the loss
                loss = loss_function(model_output.view(-1, number_of_tokens), target.view(-1))

                loss.backward()

                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                # Backpropagate the loss
                optimizer.step()

                # Reset the gradients
                optimizer.zero_grad()

                losses.append(loss.item())

            # Print the loss
            print('Epoch:', epoch, 'Loss:', np.average(losses))

        # Generate some text
        model.eval()
        tokens_to_generate = 50
        input_tensor = torch.tensor(pad_left(tokenizer.tokenize("elephants"), 21, tokenizer.character_to_token('<PAD>')), dtype=torch.long)

        generated_text = model.generate(start_tokens=input_tensor, seq_len=tokens_to_generate, eos_token=None)
        generated_text = generated_text[0].tolist()
        print('Generated text:', ''.join([tokenizer.token_to_character(token) for token in generated_text]))


def pad_left(sequence, final_length, padding_token):
    return [padding_token] * (final_length - len(sequence)) + sequence


Runner().run()
