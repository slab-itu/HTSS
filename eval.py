import argparse
import os
import time

import pandas as pd
import torch as T
from rouge import Rouge

import pp
from beam_search import *
from data_util import config
from model import Encoder_Decoder_Model
from pp import Vocab
from train_util import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###################################

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences
###################################

if T.cuda.is_available():
    T.cuda.manual_seed_all(123)


class Evaluate(object):
    def __init__(self, opt):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.opt = opt
        time.sleep(5)

    def setup_valid(self):
        self.model = Encoder_Decoder_Model()
        self.model = get_cuda(self.model)
        if T.cuda.is_available():
            checkpoint = T.load(os.path.join(config.save_model_path, self.opt.load_model))
        else:
            checkpoint = T.load(os.path.join(config.save_model_path, self.opt.load_model))
        self.model.load_state_dict(checkpoint["model_dict"])

    def print_original_predicted(self, decoded_sents, ref_sents, article_sents, loadfile):
        filename = "score_files/txt/test_" + loadfile.split(".")[0] + ".txt"
        with open(os.path.join("", filename), "w") as f:
            for i in range(len(decoded_sents)):
                f.write("article: " + article_sents[i] + "\n")
                f.write("ref: " + ref_sents[i] + "\n")
                f.write("dec: " + decoded_sents[i] + "\n\n")

    def evaluate_batch(self, data_set, print_sents=False):
        print("preparing to evaluate the model....")
        self.setup_valid()
        print("Evaluating the model ....", self.opt.load_model)
        start_id = self.vocab.word2id(START_DECODING)
        end_id = self.vocab.word2id(STOP_DECODING)
        unk_id = self.vocab.word2id(UNKNOWN_TOKEN)
        decoded_sents = []
        ref_sents = []
        article_sents = []
        rouge = Rouge()
        # while batch is not None:
        prev = 0
        batch_count = 0
        for next in range(config.batch_size, len(data_set), config.batch_size):
            batch_count += 1
            print("processing batch:", batch_count)
            batch1 = pp.get_next_batch(data_set, prev, next)
            prev = next
            ## batch = self.batcher.next_batch()
            batch = Batch(example_list=batch1, vocab=pp.vocab, batch_size=config.batch_size)
            enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e = get_enc_data(batch)
            with T.autograd.no_grad():
                enc_batch = self.model.embeds(enc_batch)
                enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)
            # -----------------------Summarization----------------------------------------------------
            with T.autograd.no_grad():
                pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab,
                                       self.model, start_id, end_id, unk_id)

            for i in range(len(pred_ids)):
                decoded_words = pp.outputids2words(pred_ids[i], self.vocab, batch.art_oovs[i])
                # print("decoded words are here....")
                if len(decoded_words) < 2:
                    decoded_words = "xyz"
                else:
                    decoded_words = " ".join(decoded_words)
                if (print_sents):
                    print("decoded words: ", decoded_words)
                    print("original words: ", batch.original_abstracts[i])
                decoded_sents.append(decoded_words)
                abstract = batch.original_abstracts[i]
                article = batch.original_articles[i]
                ref_sents.append(abstract)
                article_sents.append(article)
        load_file = self.opt.load_model

        if print_sents:
            print("print sentence param is :", print_sents)
            self.print_original_predicted(decoded_sents, ref_sents, article_sents, load_file)
        scores = rouge.get_scores(decoded_sents, ref_sents, avg=True)
        sari_score = pp.compute_sari(source=article_sents, refs=ref_sents, decoded_sents=decoded_sents).item()
        if self.opt.task == "test":
            print(load_file, "scores:", scores)
            print("SARI score:", sari_score)
            return scores, sari_score
        else:
            print("Evaluation results....")
            rouge_1 = scores["rouge-1"]["f"]
            print(load_file, "rouge_1:", "%.4f" % rouge_1, "SARI:", "%.4f" % sari_score)
            return scores, sari_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="validate", choices=["validate", "test"])
    parser.add_argument("--start_from", type=str, default="020.tar")
    parser.add_argument("--load_model", type=str, default="046.tar")

    opt = parser.parse_args()
    data_set = pd.read_csv(config.test_data_path, encoding="ISO-8859-1")

    eval_processor = Evaluate(opt)
    if opt.task == "validate":
        saved_models = os.listdir(config.save_model_path)
        saved_models.sort()
        file_idx = saved_models.index(opt.start_from)
        saved_models = saved_models[file_idx:]
        rg1_score = []
        rg2_score = []
        rgL_score = []
        sar_score = []
        for f in saved_models:
            opt.load_model = f
            rg, sari = eval_processor.evaluate_batch(data_set=data_set, print_sents=True)
            rg1_score.append(rg["rouge-1"]["f"])
            rg2_score.append(rg["rouge-2"]["f"])
            rgL_score.append(rg["rouge-l"]["f"])
            sar_score.append(sari)
        print("rg1..............................")
        print(rg1_score)
        print("rg2................................")
        print(rg2_score)
        print("rg L....................................")
        print(rgL_score)
        print("sari...............................")
        print(sar_score)
    else:  # test
        rg, sari = eval_processor.evaluate_batch(data_set=data_set, print_sents=False)

######################
#### best scores
scores = {'rouge-1': {'f': 0.21781688458084078, 'p': 0.2961132698684665, 'r': 0.17915779006620344},
          'rouge-2': {'f': 0.03631698338653126, 'p': 0.0434852194176788, 'r': 0.03220525226508185},
          'rouge-l': {'f': 0.16762338538057953, 'p': 0.25479104517870454, 'r': 0.15320931531922163}}

SARI_score = 38.36989974975586

scores = {'rouge-1': {'f': 0.21839595106871204, 'p': 0.29277623937500746, 'r': 0.1803141040878516},
          'rouge-2': {'f': 0.037330271593543136, 'p': 0.04432847275135537, 'r': 0.033181430107819716},
          'rouge-l': {'f': 0.1688190071485557, 'p': 0.25129856657950433, 'r': 0.15418418042442442}}
SARI_score = 38.41526794433594

# finally tested model
# 046.tar scores
score = {'rouge-1': {'f': 0.21870449124455865, 'p': 0.2888385919903398, 'r': 0.18219781990871384},
         'rouge-2': {'f': 0.03929273962196897, 'p': 0.046068057913773726, 'r': 0.03514424243826909},
         'rouge-l': {'f': 0.17047006666914694, 'p': 0.24840616086662534, 'r': 0.15583127819083578}}
SARI_score = 38.52761459350586

########################################################
## model = 019.tar
beta = 0.1
CSS1 = 30.229

scores = {'rouge-1': {'f': 0.2430725691706964, 'p': 0.31718626531602667, 'r': 0.20155058313340346},
          'rouge-2': {'f': 0.05204433147803612, 'p': 0.061717609674972966, 'r': 0.04586662007937396},
          'rouge-l': {'f': 0.1896663743792499, 'p': 0.27098223861501886, 'r': 0.17240840224159654}}
SARI_score = 39.98530197143555

## model 014.tar
beta = 0.3
CSS1 = 30.696

scores = {'rouge-1': {'f': 0.24939046570573148, 'p': 0.32748017936105933, 'r': 0.20564528849702687},
          'rouge-2': {'f': 0.05335451336453882, 'p': 0.06455475282468259, 'r': 0.046321975506820434},
          'rouge-l': {'f': 0.19580107077063935, 'p': 0.2832182931647633, 'r': 0.17748118082026332}}
SARI_score = 39.91069030761719

# model 027.tar
scores = {'rouge-1': {'f': 0.22779332836142216, 'p': 0.2628128283823688, 'r': 0.20456833569511956},
          'rouge-2': {'f': 0.04029929749057473, 'p': 0.04361138273757405, 'r': 0.03790945935893431},
          'rouge-l': {'f': 0.19131769001795973, 'p': 0.22974375450392792, 'r': 0.17873768090462508}}
SARI_score = 38.091392517089844
