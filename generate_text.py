import sys
sys.path.append('..')
from rnnlm_gen import RnnlmGen
from dataset import aozora_bunko
from dataset import ptb

corpus, word_to_id, id_to_word = aozora_bunko.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RnnlmGen(vocab_size=vocab_size)
model.load_params('./models/AozoraRnnlm.pkl')

# start文字とskip文字の設定
start_word = 'あなた'
start_id = word_to_id[start_word]
skip_words = []
skip_ids = [word_to_id[w] for w in skip_words]

# 文章生成
word_ids = model.generate(start_id, skip_ids)
eos_id = word_to_id['<eos>']
txt = ''.join([id_to_word[i] if i != eos_id else "。\n" for i in word_ids])
txt = txt.replace('<eos>', '.\n')
print(txt)
