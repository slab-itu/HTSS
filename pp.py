# import data_util.config as config
import csv

# import spacy
# nlp = spacy.load("en_core_web_sm")
import nltk
import pandas as pd
import torch as T

import data_util.config as config


def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


#######################################################
def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError(
                    'Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                        i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


#############################################################
### Batch


#############################################################
class Vocab(object):
    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r', encoding='utf-8') as vocab_f:
            for line in vocab_f:
                pieces = line.split("#_#_#")
                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file at line : %s\n' % pieces)
                    continue
                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    print("w = ", w)
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    # print ("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break

        # print ("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))

    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        return self._count

    def write_metadata(self, fpath):
        print("Writing word embedding metadata file to %s..." % (fpath))
        with open(fpath, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(self.size()):
                writer.writerow({"word": self._id_to_word[i]})


###############################################
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences


def article2ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


################################################

def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


#######################################################
class Example(object):
    def __init__(self, article, summary, vocab):
        # Get ids of special tokens
        start_decoding = vocab.word2id(START_DECODING)
        stop_decoding = vocab.word2id(STOP_DECODING)

        # Process the article
        article_words = nltk.word_tokenize(article)
        # article_words = article.split()
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]
        self.enc_len = len(article_words)  # store the length after truncation but before padding
        # list of word ids; OOVs are represented by the id for UNK token
        self.enc_input = [vocab.word2id(w) for w in article_words]
        # Process the abstract
        summary_words = nltk.word_tokenize(summary)
        # list of word ids; OOVs are represented by the id for UNK token
        abs_ids = [vocab.word2id(w) for w in summary_words]

        # Get the decoder input sequence and target sequence
        self.dec_input, _ = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id;
        # also store the in-article OOVs words themselves
        self.enc_input_extend_vocab, self.article_oovs = article2ids(article_words, vocab)

        # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
        abs_ids_extend_vocab = abstract2ids(summary_words, vocab, self.article_oovs)

        # Get decoder target sequence
        _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding,
                                                    stop_decoding)

        # Store the original strings
        self.original_article = article
        self.original_abstract = summary
        # self.original_abstract_sents = abstract_sentences

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        while len(self.enc_input_extend_vocab) < max_len:
            self.enc_input_extend_vocab.append(pad_id)


################################################
vocab = Vocab(config.vocab_path, config.vocab_size)


def get_next_batch(data_set, prev, batch_size):
    batch = data_set.values[prev:batch_size]
    if (len(batch) != config.batch_size):
        return 0
    ex_batch = []
    for row in batch:
        simple_title = row[1]
        # original_title = row[3]
        article_path, summary = row[4], row[2]
        summary = simple_title + " " + summary
        # print(summary)
        article_path = article_path.replace(".xml", ".txt")
        article_path = "data/" + article_path
        article_object = open(article_path, 'r', encoding='ISO-8859-1')
        article = article_object.readlines()
        article = [
            l.replace('  \n', '').replace('  #@NEW_LINE#@#', '').replace('title', '').replace('Abstract', '').replace(
                'Introduction', '') for l in article]
        article = [l for l in article if len(l) > 2]
        article = ''.join(article[1:5])
        # doc = nlp(article)
        sent = nltk.sent_tokenize(article)[:config.n_input_sents]
        # sent = [token.text for token in doc.sents][:config.n_input_sents]
        article = [val.lower() for sublist in sent for val in nltk.word_tokenize(sublist)]
        # article = [val.text.lower() for sublist in sent for val in nlp(sublist)]
        ##########################################################3
        # doc = nlp(summary)
        # sent = [token.text for token in doc.sents][:config.n_output_sents]
        sent = nltk.sent_tokenize(summary)
        sent = [s for s in sent if
                "view more credit" not in s.lower() and 'image' not in s.lower() and 'credit' not in s.lower()][
               :config.n_output_sents]
        # summary = [val.text.lower() for sublist in sent for val in nlp(sublist)]
        summary = [val.lower() for sublist in sent for val in nltk.word_tokenize(sublist)]
        if (len(article) > config.max_enc_steps):
            article = article[:config.max_enc_steps]
        if (len(summary) > config.max_dec_steps):
            summary = summary[:config.max_dec_steps]
        article = ' '.join(article)
        summary = ' '.join(summary)
        if ("view more credit".lower() in summary):
            print("yes")
        example = Example(article, summary.strip(), vocab)
        ex_batch.append(example)
        # print(example)
        # print("this is pair 1: ", article, "\nthis is pair 2:", summary_sentences, "\n")
    # btch = Batch(ex_batch, vocab, batch_size=len(ex_batch))
    return ex_batch


def compute_easy(decoded_sents, dictionary):
    count_hard = 1
    for sent in decoded_sents:
        sent_score = 0
        for w in sent.split(" "):
            if w in dictionary:
                sent_score += dictionary[w]
        sent_score_avg = sent_score / len(sent.split(" "))
        count_hard += sent_score_avg
    score = count_hard / len(decoded_sents)
    score = get_cuda(T.FloatTensor([score]))
    return score


def compute_sari(source, decoded_sents, refs):
    from easse.sari import corpus_sari
    score_list = []
    for source_sent, decoded, ref in zip(source, decoded_sents, refs):
        decoded = [decoded]
        ref = [[ref]]
        source_sent = [source_sent]
        # print("decoded:", decoded)
        # print("Ref:", ref)
        sari_score = corpus_sari(orig_sents=source_sent, sys_sents=decoded, refs_sents=ref)
        score_list.append(sari_score)
    # sari_score = corpus_sari(orig_sents=refs, sys_sents=decoded_sents, refs_sents=refs)
    score = get_cuda(T.FloatTensor(score_list)).sum() / len(score_list)
    return score


def words_tagged(path):
    tagged = pd.read_csv(path, encoding='ISO-8859-1')
    dictionary = {}
    faults = 0
    correctly_skipped = 0

    for line in tagged.values:
        w = line[0]
        tag = line[1]
        if (w not in dictionary):
            dictionary[w] = tag
        else:
            # print("skipping word:", w, "new_tag:", tag, " Old tag: ", dictionary[w])
            if (tag != dictionary[w]):
                faults += 1
            else:
                correctly_skipped += 1
    return dictionary
#######################################################################
# data_set = pd.read_csv(config.path, encoding="ISO-8859-1")
#
# b = get_next_batch(data_set, 0, config.batch_size)
#
# prev = 0
# batch_count = 0
#
# for next in range(config.batch_size, len(data_set), config.batch_size):
#     batch_count+=1
#     batch  = get_next_batch(data_set, prev, next)
#     prev = next
#     break
#######################################################################
