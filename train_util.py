import numpy as np
import torch as T

import pp
from data_util import config as config


# from data_util import config as config


def get_cuda(tensor):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"  # Set cuda device
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def get_enc_data(batch):
    # batch = Batch(example_list=batch1, vocab=pp.vocab, batch_size=config.batch_size)
    batch_size = len(batch.enc_lens)
    enc_batch = T.from_numpy(batch.enc_batch).long()
    enc_padding_mask = T.from_numpy(batch.enc_padding_mask).float()
    enc_lens = batch.enc_lens
    ct_e = T.zeros(batch_size, 2 * config.hidden_dim)
    enc_batch = get_cuda(enc_batch)
    enc_padding_mask = get_cuda(enc_padding_mask)
    ct_e = get_cuda(ct_e)
    enc_batch_extend_vocab = None
    if batch.enc_batch_extend_vocab is not None:
        enc_batch_extend_vocab = T.from_numpy(batch.enc_batch_extend_vocab).long()
        enc_batch_extend_vocab = get_cuda(enc_batch_extend_vocab)
    extra_zeros = None
    if batch.max_art_oovs > 0:
        extra_zeros = T.zeros(batch_size, batch.max_art_oovs)
        extra_zeros = get_cuda(extra_zeros)
    return enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e


def get_dec_data(batch):
    # batch = Batch(example_list=batch1, vocab=pp.vocab, batch_size=config.batch_size)
    dec_batch = T.from_numpy(batch.dec_batch).long()
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    dec_lens = T.from_numpy(batch.dec_lens).float()
    target_batch = T.from_numpy(batch.target_batch).long()
    dec_batch = get_cuda(dec_batch)
    dec_lens = get_cuda(dec_lens)
    target_batch = get_cuda(target_batch)
    return dec_batch, max_dec_len, dec_lens, target_batch


class Batch(object):
    def __init__(self, example_list, vocab, batch_size):
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(pp.PAD_TOKEN)  # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list)  # initialize the input to the encoder
        self.init_decoder_seq(example_list)  # initialize the input and targets for the decoder
        self.store_orig_strings(example_list)  # store the original strings

    def init_encoder_seq(self, example_list):
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        # For pointer-generator mode, we need to store some extra info
        # Determine the max number of in-article OOVs in this batch
        self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
        # Store the in-article OOVs themselves
        self.art_oovs = [ex.article_oovs for ex in example_list]
        # Store the version of the enc_batch that uses the article OOV ids
        self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        for i, ex in enumerate(example_list):
            self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def init_decoder_seq(self, example_list):
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        # self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            # for j in range(ex.dec_len):
            #   self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        self.original_articles = [ex.original_article for ex in example_list]  # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list]  # list of lists
        # self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list]  # list of list of lists
