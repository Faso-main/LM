import gensim
import pymorphy2

model = gensim.models.KeyedVectors.load_word2vec_format("model.bin", binary=True)
model.init_sims(replace=True)

morph = pymorphy2.MorphAnalyzer()