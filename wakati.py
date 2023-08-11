#!/usr/bin/env python
# coding: utf-8

# In[16]:


import MeCab as mc
import re
import os
from bs4 import BeautifulSoup
from urllib import request
import glob
from pathlib import Path


# In[6]:


def get_author_id_dict(url="https://www.aozora.gr.jp/index_pages/person_all.html"):
    '''
    機能：著者名と著者IDの辞書を取得する関数
    引数：青空文庫の作家リストページのURL(省略可)
    返り値：著者名と著者IDの辞書
    '''
    response = request.urlopen(url)
    soup = BeautifulSoup(response)
    response.close()

    author_id = {}
    elms = soup.select('li a')
    for e in elms:
        href = e['href']
        idx = re.findall('person([0-9]*)\.html.*',href)
        author_id[e.text.replace(' ','')] = f'{int(idx[0]):0=6}'
    return author_id

author_id_dict = get_author_id_dict()

import pickle
with open('author_id_dict.pkl','wb') as f:
    pickle.dump(author_id_dict,f)


# In[7]:


def get_local_author_id_dict():
    import pickle
    with open('author_id_dict.pkl','rb') as f:
        author_id_dict = pickle.load(f)
    return author_id_dict
get_local_author_id_dict()


# In[17]:


def text_pre_processing(txt):# 本文の前処理
    '''
    機能：文字列(テキスト)の不要な部分(本文以外の文書や「」など)を削除する関数
    変数：前処理したい文字列
    返り値：処理した文字列
    '''
    txt = txt.split('底本')[0]
    txt = txt.replace('｜','')
    txt = txt.replace('／','').replace('＼','').replace('―','')
    txt = re.sub('《.*?》','', txt)
    txt = re.sub('［＃中見出し］.*?［＃中見出し終わり］','', txt)
    txt = re.sub('［＃.*?］','', txt)
    txt = txt.replace('「', '').replace('」', '').replace('『','').replace('』','')
    txt = txt.replace('\r','').replace('\n','').replace('\u3000', '')
    txt = re.sub('([。！？])', r'\1\n', txt)
    return txt

def make_alltext_of_specified_author(author):
    '''
    機能：指定した著者名の全作品を文字列として結合し、ファイルに保存する
    変数：著者名を名字と名前を続く形で与える ex)「夏目漱石」「太宰治」
    返り値：結合した文字列(著者の全作品)
    '''
    
    all_filename_list = glob.glob(f'/Users/akimotokazuki/cards/{author_id_dict[author]}'+r'/files/*/*.txt')
    file_all_content = ''
    file_stem_list = []
    for file in all_filename_list:
        file_id = Path(file).stem.split('_')[0]
        with open(file,'rb') as f:
            file_content = f.read().decode('shiftjisx0213')
        try:
            file_content = re.split('-{10,}',file_content)[2]
            if  file_id in file_stem_list:
                continue
            file_content = text_pre_processing(file_content)
            file_all_content += file_content
            file_stem_list.append(file_id)
        except Exception as e: # 例外処理。置換に失敗したファイルの内容は捨てる
            pass
        
    with open(f'word2vec_data/Git{author}.txt','w') as f:
        print(file_all_content, file = f)   
    
    return file_all_content

def create_segmented_list_of_sentences_by_author(text_data,stopword=True): 
    '''
    機能：文章をわかち書きし、Stopword(「の」や「が」など)を文章から省く関数
    変数：処理したい文字列、定義したストップワード
    返り値：処理した文字列
    '''
    
    dic_neo = ' -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd' 
    wakati = mc.Tagger("-Owakati"+dic_neo)
    
    if stopword:
        pass
        # stopwords = create_japanese_stopword()
    else:
        stopwords = []
    
    text_wakati_data = [wakati.parse(i).split() for i in text_data.split('\n')]
    text_data_result = []
    
    for s in text_wakati_data:
        s_list = []
        for w in s:
            if w not in stopwords:
                s_list.append(w)
        text_data_result.append(s_list)
                    
    return text_data_result


# In[18]:


alltext_of_author = make_alltext_of_specified_author("太宰治")
text_data_result = create_segmented_list_of_sentences_by_author(alltext_of_author, stopword=False)
print(len(text_data_result))


# In[10]:


# 訓練データ
train_file_content = ""
for row in text_data_result[:int(len(text_data_result)/5*3)]:
        for word in row:
            train_file_content += word + " "
        train_file_content += '\n'
print(train_file_content[:100])
print(len(train_file_content.split(" ")))


# In[11]:


# テストデータ
test_file_content = ""
for row in text_data_result[int(len(text_data_result)/5*4):int(len(text_data_result)/5*4.5)]:
        for word in row:
            test_file_content += word + " "
        test_file_content += '\n'
print(test_file_content[:100])
print(len(test_file_content.split(" ")))


# In[12]:


# 評価データ
valid_file_content = ""
for row in text_data_result[int(len(text_data_result)/5*4.5):]:
        for word in row:
            valid_file_content += word + " "
        valid_file_content += '\n'
print(valid_file_content[:100])
print(len(valid_file_content.split(" ")))


# In[13]:


# わかち書きテキストファイル生成
train_file_name = 'train_dazai.txt'  
with open(train_file_name, 'w', encoding='utf-8') as f:
    f.write(train_file_content)

test_file_name = 'test_dazai.txt'  
with open(test_file_name, 'w', encoding='utf-8') as f:
    f.write(test_file_content)
    
valid_file_name = 'valid_dazai.txt' 
with open(valid_file_name, 'w', encoding='utf-8') as f:
    f.write(valid_file_content)

