# coding: utf-8
import sys
sys.path.append('..')
from rnnlm_gen import RnnlmGen
from rnnlm_gen import BetterRnnlmGen
from dataset import ptb
from dataset import aozora_bunko


# corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus, word_to_id, id_to_word = aozora_bunko.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

# model = RnnlmGen(vocab_size=vocab_size)
model = BetterRnnlmGen(vocab_size=vocab_size)
model.load_params('../ch06/BetterRnnlm.pkl')


# start文字とskip文字の設定
start_word = 'あなた'
start_id = word_to_id[start_word]
# 英語文章生成
# skip_words = ['N', '<unk>', '$']
# skip_ids = [word_to_id[w] for w in skip_words]
# word_ids = model.generate(start_id, skip_ids)
# txt = ' '.join([id_to_word[i] for i in word_ids])
# txt = txt.replace(' <eos>', '.\n')

# 日本語文章生成
skip_words = []  # ★前処理していないのでskip文字はなし
skip_ids = [word_to_id[w] for w in skip_words]
word_ids = model.generate(start_id, skip_ids)
# ★日本語なので空白なしで連結、<eos>は句点＋改行に置換
eos_id = word_to_id['<eos>']
txt = ''.join([id_to_word[i] if i != eos_id else '。\n' for i in word_ids])
txt = txt.replace('\n。\n', '\n')
print(txt)
