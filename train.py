
import argparse
import time
import matplotlib.pyplot as plt
import torch as T
import torch.nn.functional as F
import torch.nn as nn
from numpy import random
from rouge import Rouge
# from torch.distributions import Categorical
from pp import Vocab
from model import Encoder_Decoder_Model
from data_util import config
from train_util import *
import pandas as pd
import pp
import os
from GPUtil import showUtilization as gpu_usage
from beam_search import beam_search

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # setting trace back for more clear error
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set cuda device

random.seed(123)
T.manual_seed(123)
if T.cuda.is_available():
    T.cuda.manual_seed_all(123)

#############################
# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences
############################


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        # self.batcher = Batcher(config.train_data_path, self.vocab, mode='train', batch_size=config.batch_size, single_pass=False)
        self.opt = opt
        self.start_id = self.vocab.word2id(START_DECODING)
        self.end_id = self.vocab.word2id(STOP_DECODING)
        self.pad_id = self.vocab.word2id(PAD_TOKEN)
        self.unk_id = self.vocab.word2id(UNKNOWN_TOKEN)
        self.model_enc_dec = Encoder_Decoder_Model()
        self.optimizer = T.optim.Adam(self.model_enc_dec.parameters(), lr=config.lr)
        self.model_enc_dec = get_cuda(self.model_enc_dec)
        self.cant_do_more = 0
        self.word_tag=pp.words_tagged(config.tag_path)
        ####################################################
        if self.opt.load_model is not None:
            self.start_iter = self.load_model()
        else:
            self.start_iter =0
        # if self.opt.new_lr is not None:
        #     self.optimizer = T.optim.Adam(self.model_enc_dec.parameters(), lr=self.opt.new_lr)
        ####################################################

    def save_model(self, iter):
        save_path = config.save_model_path + "/%03d.tar" % iter
        T.save({ "iter": iter + 1,  "model_dict": self.model_enc_dec.state_dict(),
                 "optimizer_dict": self.optimizer.state_dict() }, save_path)

    def train_one_batch(self, batch, iter):
        enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, context = get_enc_data(batch)
        enc_batch = self.model_enc_dec.embeds(enc_batch)  # Get embeddings for encoder input
        enc_out, enc_hidden = self.model_enc_dec.encoder(enc_batch, enc_lens)

        # -------------------------------Summarization-----------------------
        cant_do_more=0

        if cant_do_more==0:  # perform MLE training
            ####################################################################################
            # Get input and target batchs for training decoder
            dec_batch, max_dec_len, dec_lens, target_batch = get_dec_data(batch)
            step_losses = []
            # Decoder hidden states
            s_t = (enc_hidden[0], enc_hidden[1])
            # Input to the decoder
            x_t = get_cuda(T.LongTensor(len(enc_out)).fill_(self.start_id))
            # Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
            prev_s = None
            # Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
            sum_temporal_srcs = None
            cant_do_more = 0
            ct_e = context
            for t in range(min(max_dec_len, config.max_dec_steps)):
                # Probabilities indicating whether to use ground truth labels instead of previous decoded tokens
                use_gound_truth = get_cuda(T.rand(len(enc_out)) > 0.25).long()
                # Select decoder input based on use_ground_truth probabilities
                x_t = use_gound_truth * dec_batch[:, t] + (1 - use_gound_truth) * x_t
                x_t = self.model_enc_dec.embeds(x_t)
                final_dist, s_t, ct_e, sum_temporal_srcs, prev_s = \
                    self.model_enc_dec.decoder(x_t, s_t, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s)
                target = target_batch[:, t]
                log_probs = T.log(final_dist + config.eps)
                # step_loss = F.cross_entropy(log_probs, target, reduction='none', ignore_index=self.pad_id)
                step_loss = F.nll_loss(log_probs, target, reduction="none", ignore_index=self.pad_id)
                step_losses.append(step_loss)
                # Sample words from final distribution which can be used as input in next time step
                nans = T.sum(T.isnan(final_dist)).item()
                if (nans > 0):
                    print("Not a valid probability distribution nans values")
                    # self.save_model()
                    self.save_model(self.start_iter)
                    cant_do_more = 1
                    return 0.0, 0.0, cant_do_more
                    break
                    # final_dist[T.isnan(final_dist)] = 1
                # final_dist[final_dist < 0] = 1
                x_t = T.multinomial(final_dist, 1).squeeze()
                is_oov = (x_t >= config.vocab_size).long()  # Mask indicating whether sampled word is OOV
                x_t = (1 - is_oov) * x_t.detach() + (is_oov) * self.unk_id  # Replace OOVs with [UNK] token

            losses = T.sum(T.stack(step_losses, 1), 1)  # unnormalized losses for each example in the batch; (batch_size)
            batch_avg_loss = losses / config.batch_size  # Normalized losses; (batch_size)
            mle_loss = T.mean(batch_avg_loss)  # Average batch loss
            #################################################
            simplification_loss = get_cuda(T.FloatTensor([0.0]))
            if (self.opt.simplification and self.cant_do_more==0):
                   with T.autograd.no_grad():
                       ct_e = context
                       pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab,
                                              self.model_enc_dec, self.start_id, self.end_id, self.unk_id)
                       decoded_sents = []
                       ref_sents = []
                       article_sents=[]
                       for i in range(len(pred_ids)):
                           decoded_words = pp.outputids2words(pred_ids[i], self.vocab, batch.art_oovs[i])
                           if len(decoded_words) < 2:
                               decoded_words = "xyz"
                           else:
                               decoded_words = " ".join(decoded_words)
                               # print("decoded words: ", decoded_words)
                               # print("original words: ", batch.original_abstracts[i])
                           decoded_sents.append(decoded_words)
                           ref_sents.append(batch.original_abstracts[i])
                           article_sents.append(batch.original_articles[i])
                       # simplification_loss = pp.compute_easy(decoded_sents, self.word_tag)
                       # print(simplification_loss)
                       simplification_score = pp.compute_sari(article_sents, decoded_sents, ref_sents)
                       simplification_loss = 100-simplification_score
            #################################################
            # print("batch completed calling loss.backward()")
            self.optimizer.zero_grad()
            (mle_loss + config.beta * simplification_loss).backward()
            # mle_loss.backward()
            self.optimizer.step()

        else:
            mle_loss = get_cuda(T.FloatTensor([0]))
            simplification_loss = get_cuda(T.FloatTensor([0]))
        # if(cant_do_more==0):
            # (self.opt.mle_weight * mle_loss + self.opt.rl_weight * rl_loss).backward()
            # self.optimizer.zero_grad()
            # mle_loss.backward()
            # self.optimizer.step()
        return mle_loss.item(), simplification_loss.item(), cant_do_more

    def load_model(self):
        load_model_path = os.path.join(config.save_model_path, self.opt.load_model)
        checkpoint = T.load(load_model_path)
        start_iter = checkpoint["iter"]
        self.model_enc_dec.load_state_dict(checkpoint["model_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_dict"])
        print("Loaded model at " + load_model_path)
        return start_iter

    def forward(self, batch):
        if (self.cant_do_more == 0):
            mle_loss, simp_loss, self.cant_do_more = self.train_one_batch(batch, self.start_iter)
            return mle_loss, simp_loss
        else:
            print("cant_do_more iterations .... stopping here to avoid nans")
            simp_loss=T.tensor(0.1)
            mle_loss = T.tensor(0.1)
            return mle_loss, simp_loss

def run_train():
    data_set = pd.read_csv(config.train_data_path, encoding="ISO-8859-1")  # [:400]
    model = Model(opt)
    # model2 = nn.DataParallel(model, device_ids=[0, 1])
    # model2 = get_cuda(model2)
    plot_loss = []
    while model.start_iter <= config.max_iterations and model.cant_do_more==0:
        # print("Epoch : %i, and  %.1f percent completed" % ( iter, iter*100/config.max_iterations))
        start_time =  time.time()
        prev = 0
        batch_count = 0
        mle_total = 0
        simplification_total =0
        count=0
        for next in range(config.batch_size, len(data_set), config.batch_size):
            batch_count+=1
            batch  = pp.get_next_batch(data_set, prev, next)
            batch = Batch(example_list=batch, vocab=pp.vocab, batch_size=config.batch_size)
            prev = next
            try:
                if (model.cant_do_more == 0):
                    mle_loss, simp_loss = model(batch)
                else:
                    print("cant_do_more iterations .... stopping here to avoid nans")
                    break
            except KeyboardInterrupt:
                print("-------------------Keyboard Interrupt------------------")
                exit(0)

            mle_total += mle_loss
            simplification_total += simp_loss
            count += 1
        model.start_iter += 1
        if model.start_iter % 1 == 0:
            mle_avg = mle_total / count
            simpl_avg = simplification_total / count
            plot_loss.append(mle_avg)
            print("Epoch:", model.start_iter, "MLE_Loss:", "%.3f" % (mle_avg),
                  "Simplification_Loss :", simpl_avg, "time:", (time.time()-start_time)/60, " minutes" )
            count = mle_total = r_total = 0
        if model.start_iter % 1 ==0:
            model.save_model(model.start_iter)
    return plot_loss




if __name__ == "__main__":
    T.cuda.empty_cache()
    print("GPU utilization after clearing the catche: ")
    gpu_usage()
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--simplification', type=bool, default=False)
    opt = parser.parse_args()
    print("\n Training the model....")
    print("intra_encoder:", config.intra_encoder, "intra_decoder:", config.intra_decoder,
          "beta:", config.beta, "simplification:", opt.simplification)
    # print("CUDA_VISIBLE_DEVICES =", os.environ['CUDA_VISIBLE_DEVICES'])
    time_1 = time.time()
    loss = run_train()
    time_2 =  time.time()
    print("Training Finished elapsed time:", (time_2-time_1)/3600, " hours")
    print("plot the loss")
    plt.plot(loss)
    plt.show()

