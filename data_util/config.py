vocab_path = "data/vocab"
path = "data/new_data_set.csv"

train_data_path = "data/train.csv"
test_data_path = "data/test.csv"
# tag_path = "data/TaggedWords.csv"
tag_path = "data/WordsTagged.csv"

beta = 0.0
n_input_sents = 5
n_output_sents = 3

# Hyperparameters
hidden_dim = 512
emb_dim = 512
batch_size = 145

# max_enc_steps = 55  # 99% of the articles are within length 55
max_enc_steps = 100  # 99% of the articles are within length 55

# max_dec_steps = 15  # 99% of the titles are within length 15
max_dec_steps = 40  # 99% of the titles are within length 15
beam_size = 8  ## 5 for train changed
min_dec_steps = 5  ## changed

# vocab_size = 64466
# vocab_size = 71297
# vocab_size = 25000
# vocab_size = 15000
# vocab_size = 80611
vocab_size = 49845

lr = 0.001

rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
eps = 1e-12
max_iterations = 120
save_model_path = "data/saved_models"
intra_encoder = True
intra_decoder = True
use_stable_softmax = False
