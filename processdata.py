""" consist routines for finding word embeddings, sentence embeddings, sentence
    preprocessing etc """

from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import tokenize
import pandas as pd
import numpy as np
import sqlite3


def get_words_data():
  """ return dictionary, lemma and embedding from words table """
  
  try:
    conn = sqlite3.connect('web.db')
    cur = conn.cursor()
    rows = cur.execute('SELECT * from dictionary')

    dictionary, lemma, embd = set(), {}, {}

    for row in rows:
      dictionary.add(row[0])
      lemma[row[0]] = row[2]
      embd[row[0]] = row[1]
    
    return dictionary, lemma, embd

  except sqlite3.Error as error:
    print(error)


dictionary, lemma, embd = get_words_data()

#Frequent Terms find on web
AVOID = {'error', 'enable', 'jump', 'rights', 'block', 'condition', 'javascript', 'fail', 'menu', 'filipino', 'register', 'site', 'request', 'dutch', 'espanol', 'reserved', 'help', 'home', 'url', 'italiano', 'page', 'navigate', 'cookie', 'browser', 'disable', 'cancel', 'unsupported', 'english', 'francais', 'login', 'privacy', 'term', 'section', 'disabled', 'skip', 'main', 'copyright', 'uses', 'navigation', 'more', 'anymore', 'log', 'open', 'homepage', 'corona', 'policy', 'content', 'terms', 'sign', 'upgrade', 'portugal', 'older', 'require', 'know', 'indonesia', 'support', 'results', 'language', 'coronavirus', 'best', 'load', 'deutsch', 'cookies'}

DIMENSION = 50

def get_embedding(word):
  try:
    return np.frombuffer(embd[word], dtype=float)
  except:
    return None


def average_word_embedding(words):
  """ param -: list of words 
      return -: average embedding of list of words as numpy array or None in case of failure"""

  res = np.zeros(DIMENSION)
  tot_words = 0
  
  for word in words:
    word_embd = get_embedding(word)
    if word_embd is not None:
      tot_words += 1
      res = np.add(res, word_embd)
  
  if tot_words == 0: return None
  else: return res/tot_words


def sent_embedding(sentence):
  """ param -: space seprated string 
      return -: sentence embedding as numpy array or None in case of failure """

  return average_word_embedding(sentence.split())


def sent_embedding_in_bytes(sentence):
  """ param -: space seprated string  
      return -: sentence embedding in bytes """
  try:
    return sent_embedding(sentence).tostring()
  except: 
    return None


def compress_sentence(sentence):
  """ param -: space seprated string  
      return -: compressed space seprated string in form of 'word1:freq1 word2:freq2' 
      arranged in descending order by frequency """

  freq_lst = FreqDist(sentence.split()).most_common()
  return ' '.join([row[0]+':'+str(row[1]) for row in freq_lst])


def expand_sentence(sentence):
  """ param -: compressed form of sentence
      return -: expand form of sentence """

  exp_word = lambda cmp_word: (cmp_word.split(':')[0] + ' ')*(int(cmp_word.split(':')[1])-1) + cmp_word.split(':')[0] 
  return ' '.join([exp_word(word) for word in sentence.split()])


def lemmatize(word):
  """ param -: str  
      return -: return lemmatize form of word, taking word_2_vec similarity in account to avoid change in meaning """

  return lemma.get(word, word)


def is_significant(word):
  """param -: str 
     return -: True for string b/w length 4 and 45 as well as capitalize 2 or 3 char strings """

  if not word.isalpha(): return False
  if len(word) < 2 or len(word) > 45: return False
  if (len(word) == 3 or len(word) == 2) and not word.isupper(): return False
  return True


def is_english_word(word):
  """ check whether word has all english alphabets """

  for ch in word:
    if ch < 'a' or ch > 'z':
      return False
  return True


def is_english(content):
  """ return True if more than 50% words are english """
  
  cnt = 0
  for word in content:
    if is_english_word(word): cnt += 1
  
  if cnt*2 >= len(content): return True
  else: return False


def preprocess(content):
  """  params -: raw text scrapped from website
       return -: return list of words after:    
                1) tokenization
                2) remove stopwords and some insignificant words
                3) convert in lowercase 
                4) lemmatize 
                5) Remove common web terms """

  content = tokenize(content, deacc=True)
  content = list(filter(is_significant, content))
  content = [token.lower() for token in content]
  MIN_WORDS = 30  #minimum words needed to decide whether site is english or not
  if len(content) > MIN_WORDS and not is_english(content): return ['invalidcontentfound']   #signal for non_engish site 
  content = [lemmatize(token) for token in content if token not in STOPWORDS and token in dictionary]
  content = [token for token in content if token not in AVOID]
  return content