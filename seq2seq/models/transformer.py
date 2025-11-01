import math
import torch
import torch.nn as nn
from seq2seq import utils
from seq2seq.models import register_model, register_model_architecture
from seq2seq.models import Seq2SeqModel, Seq2SeqEncoder, Seq2SeqDecoder
import sentencepiece as spm


class SinusoidPositionalEncoding(nn.Module):
    # Source: https://medium.com/@uzbrainai/transformers-and-sequence-order-sinusoidal-vs-learned-positional-embeddings-b3802df5ce24
    def __init__(self, token_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, token_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(
            0, token_dim, 2).float() * -(math.log(10000.0) / token_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

    # Ensure compatibility with other parts of the code
    def size(self, x):
        return self.pe.size(x)


@register_model('transformer')
class TransformerModel(Seq2SeqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        # self.encoder = encoder
        # self.decoder = decoder

    @staticmethod
    def add_args(parser):
        """ Add model-specific arguments to the parser. """
        parser.add_argument('--encoder-embed-path', type=str, help='Path to pre-trained encoder embeddings')
        parser.add_argument('--decoder-embed-path', type=str, help='Path to pre-trained decoder embeddings')
        # Add any additional arguments specific to the transformer model here
        parser.add_argument('--encoder-dropout', type=float, default=0.0, help='dropout probability for encoder layers')
        parser.add_argument('--decoder-dropout',type=float, default=0.0,help='dropout probability for decoder layers')
        
        parser.add_argument('--dim-embedding', type=int, default=512, help='embedding dimension for both encoder and decoder')
        parser.add_argument('--attention-heads', type=int, default=8, help='number of attention heads')
        parser.add_argument('--dim-feedforward-encoder', type=int, default=2048, help='dimension of feed-forward layers for encoder')
        parser.add_argument('--dim-feedforward-decoder', type=int, default=2048, help='dimension of feed-forward layers for decoder')
        parser.add_argument('--max-seq-len', type=int, default=128, help='maximum sequence length')
        parser.add_argument('--n-encoder-layers', type=int, default=6, help='number of encoder layers')
        parser.add_argument('--n-decoder-layers', type=int, default=6, help='number of decoder layers')
        
    @classmethod
    def build_model(cls, args, src_tokenizer, tgt_tokenizer):
        """ Constructs the model. """
        base_architecture(args)
        encoder_pretrained_embedding = None
        decoder_pretrained_embedding = None

        # Load pre-trained embeddings, if desired
        if args.encoder_embed_path:
            encoder_pretrained_embedding = utils.load_embedding(args.encoder_embed_path, src_tokenizer)
        if args.decoder_embed_path:
            decoder_pretrained_embedding = utils.load_embedding(args.decoder_embed_path, tgt_tokenizer)

        encoder = TransformerEncoder(
            src_tokenizer=src_tokenizer,
            dim_embed=args.dim_embedding,
            dropout=args.encoder_dropout,
            max_seq_len=args.max_seq_len,
            n_attention_heads=args.attention_heads,
            dim_ff=args.dim_feedforward_encoder,
            pretrained_embedding=encoder_pretrained_embedding, # currently unused
            n_encoder_layers=args.n_encoder_layers,
        )
        decoder = TransformerDecoder(
            tgt_tokenizer=tgt_tokenizer,
            dim_embed=args.dim_embedding,
            n_attention_heads=args.attention_heads,
            dropout=args.decoder_dropout,
            max_seq_len=args.max_seq_len,
            n_decoder_layers=args.n_decoder_layers,
            dim_ff=args.dim_feedforward_decoder,
            pretrained_embedding=decoder_pretrained_embedding, # currently unused
            use_cuda=args.cuda
        )
        return cls(encoder, decoder)

    def forward(self, src, src_mask, trg, trg_pad_mask):
        return self.decoder(self.encoder(src, src_mask), src_mask, trg, trg_pad_mask)

class TransformerEncoder(Seq2SeqEncoder):
    '''Encoder = token embedding + positional embedding -> a stack of N EncoderBlock -> layer norm'''
    # TODO implement usage of pretrained embeddings
    def __init__(self,
                 src_tokenizer: spm.SentencePieceProcessor,
                 dim_embed,
                 dropout,
                 max_seq_len,
                 n_attention_heads,
                 dim_ff,
                 pretrained_embedding,
                 n_encoder_layers):
        # initialize parent (but since our implementation uses a )
        super().__init__(src_tokenizer)

        self.src_vocab_size = src_tokenizer.GetPieceSize()
        
        self.dim_embed = dim_embed  # 512
        self.tok_embed = nn.Embedding(self.src_vocab_size, dim_embed)  # Vocab Dictionary size , Embed size
        self.pos_embed = SinusoidPositionalEncoding(dim_embed, max_seq_len)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(dim_embed, dropout, n_attention_heads, dim_ff) for _ in range(n_encoder_layers)])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(dim_embed)

    def forward(self, input, mask=None):
        x = self.tok_embed(input) # Vectors
        x_pos = self.pos_embed(x)  # Vectors'
        x = self.dropout(x + x_pos) # update vectors with position information
        for layer in self.encoder_blocks:
            x = layer(x, mask) # (50,512)
        
        return self.norm(x)


class EncoderBlock(nn.Module):
    '''EncoderBlock: self-attention -> position-wise fully connected feed-forward layer'''
    def __init__(self, dim_embed, dropout, n_heads, dim_ff):
        super(EncoderBlock, self).__init__()
        self.atten = MultiHeadedAttention(n_heads, dim_embed, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_embed, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim_embed)
        )
        self.residual1 = ResidualConnection(dim_embed, dropout)
        self.residual2 = ResidualConnection(dim_embed, dropout)

    def forward(self, x, mask=None):
        # self-attention
        x = self.residual1(x, lambda x: self.atten(x, x, x, mask=mask))
        # position-wise fully connected feed-forward layer
        return self.residual2(x, self.feed_forward)


class TransformerDecoder(Seq2SeqDecoder):
    '''Decoder = token embedding + positional embedding -> a stack of N DecoderBlock -> fully-connected layer'''
    # TODO implement usage of pretrained embeddings
    def __init__(self,
                 tgt_tokenizer: spm.SentencePieceProcessor,
                 dim_embed: int,
                 n_attention_heads: int,
                 dropout: float,
                 max_seq_len: int,
                 n_decoder_layers: int,
                 dim_ff: int,
                #  unused for now
                 pretrained_embedding,
                 use_cuda: bool):
        super().__init__(tgt_tokenizer)
        self.max_seq_len = max_seq_len
        self.tgt_vocab_size = tgt_tokenizer.GetPieceSize()
        self.dim_embed = dim_embed
        self.tok_embed = nn.Embedding(self.tgt_vocab_size, dim_embed)
        self.pos_embed = SinusoidPositionalEncoding(dim_embed, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.decoder_blocks = nn.ModuleList([DecoderBlock( dim_embed, n_attention_heads, dropout, dim_ff ) for _ in range(n_decoder_layers)])
        self.norm = nn.RMSNorm(dim_embed)
        self.linear = nn.Linear(dim_embed, self.tgt_vocab_size)
        self.device = torch.device("cuda" if use_cuda else "cpu")
    
    def future_mask(self, seq_len: int):
        '''mask out tokens at future positions'''
        mask = (torch.triu(torch.ones(seq_len, seq_len, requires_grad=False), diagonal=1)!=0).to(self.device)
        return mask.view(1, 1, seq_len, seq_len)

    def forward(self, encoder_out: torch.Tensor, src_mask: torch.Tensor, trg: torch.Tensor, trg_pad_mask: torch.Tensor):
        # Truncate trg to the maximum length in the batch
        max_len = self.max_seq_len  # should be 300
        if trg.size(1) > max_len:
            trg = trg[:, :max_len]
            trg_pad_mask = trg_pad_mask[:, :, :max_len]  # keep masks aligned

        seq_len = trg.size(1)
        trg_mask = torch.logical_or(trg_pad_mask, self.future_mask(seq_len))
        x = self.tok_embed(trg)
        x = self.pos_embed(x)
        x = self.dropout(x)
        for layer in self.decoder_blocks:
            x = layer(encoder_out, src_mask, x, trg_mask)
        x = self.norm(x)
        logits = self.linear(x)
        return logits

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads: int, dim_embed: int, dropout: float = 0.0):
        super(MultiHeadedAttention, self).__init__()
        #super().__init__()  python 3.x
        assert dim_embed % n_heads == 0 # check the h number
        self.d_k = dim_embed//n_heads
        self.dim_embed = dim_embed    # 512
        self.h = n_heads  # 8
        self.WQ = nn.Linear(dim_embed, dim_embed)
        self.WK = nn.Linear(dim_embed, dim_embed)
        self.WV = nn.Linear(dim_embed, dim_embed) 
        self.linear = nn.Linear(dim_embed, dim_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_query, x_key, x_value, mask=None):
        nbatch = x_query.size(0) # get batch size
        # 1) Linear projections to get the multi-head query, key and value tensors
        # x_query, x_key, x_value dimension: nbatch * seq_len * dim_embed
        # LHS query, key, value dimensions: nbatch * h * seq_len * d_k
        query = self.WQ(x_query).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        key   = self.WK(x_key).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        value = self.WV(x_value).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        # 2) Attention
        # scores has dimensions: nbatch * h * seq_len * seq_len
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(self.d_k)
        # 3) Mask out padding tokens and future tokens
        if mask is not None:
            mask.unsqueeze(dim=1)

            scores = scores.masked_fill(mask, float('-inf'))
        # p_atten dimensions: nbatch * h * seq_len * seq_len
        p_atten = torch.nn.functional.softmax(scores, dim=-1) # attention filter
        p_atten = self.dropout(p_atten)
        # x dimensions: nbatch * h * seq_len * d_k
        # print("query shape:", query.shape)
        # print("key shape:", key.shape)
        # print("value shape:", value.shape)
        # print("p_atten shape:", p_atten.shape)
        x = torch.matmul(p_atten, value)  # filtered values
        # x now has dimensions:nbatch * seq_len * dim_embed
        x = x.transpose(1, 2).contiguous().view(nbatch, -1, self.dim_embed)
        return self.linear(x) # final linear layer

class ResidualConnection(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(dim)  # (x-M)/std

    def forward(self, x, sublayer: nn.Module):
        # sublayer
        return x + self.drop(sublayer(self.norm(x)))

class DecoderBlock(nn.Module):
    ''' DecoderBlock: self-attention -> position-wise feed-forward (fully connected) layer'''
    def __init__(self, dim_embed, n_heads, dropout, dim_ff):
        super().__init__()
        self.atten1 = MultiHeadedAttention(n_heads, dim_embed)
        self.atten2 = MultiHeadedAttention(n_heads, dim_embed)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_embed, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim_embed)
        )
        self.residuals = nn.ModuleList([ResidualConnection(dim_embed, dropout) 
                                       for _ in range(3)])

    def forward(self, memory, src_mask, decoder_layer_input, trg_mask):
        x = memory  # K , V 
        y = decoder_layer_input # target /y "he"
        y = self.residuals[0](y, lambda y: self.atten1(y, y, y, mask=trg_mask)) #masked multi head attention
        # keys and values are from the encoder output
        y = self.residuals[1](y, lambda y: self.atten2(y, x, x, mask=src_mask))
        return self.residuals[2](y, self.feed_forward)


@register_model_architecture('transformer', 'transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.encoder_dropout = getattr(args, 'encoder_dropout', 0.0)
    args.decoder_dropout = getattr(args, 'decoder_dropout', 0.0)

    args.dim_embedding = getattr(args, 'dim_embedding', 512)
    args.attention_heads = getattr(args, 'attention_heads', 8)
    args.dim_feedforward_encoder = getattr(args, 'dim_feedforward_encoder', 2048)
    args.dim_feedforward_decoder = getattr(args, 'dim_feedforward_decoder', 2048)
    args.max_seq_len = getattr(args, 'max_seq_len', 512)
    args.n_encoder_layers = getattr(args, 'n_encoder_layers', 6)
    args.n_decoder_layers = getattr(args, 'n_decoder_layers', 6)
