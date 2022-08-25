## Imports
from transformers import (
    GPT2LMHeadModel, pipeline, GPT2TokenizerFast, 
    AutoTokenizer, AutoModel
    )
import torch
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity 
import nlpaug.augmenter.word as naw
from statistics import mean 
import numpy as np 
import pandas as pd 
import gensim
import gensim.downloader
import time
from helper import (
    clean, strip_punc, seperate_punct,
    pos, is_replacable
)


# ------------------------------ GPT2 Augmentation ---------------------------- #
def load_GPT(model_name):
  model = GPT2LMHeadModel.from_pretrained(model_name)
  tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
  generation_pipeline = pipeline("text-generation",model=model,tokenizer=tokenizer)
  return model , tokenizer , generation_pipeline

def GPT(model,tokenizer , generation_pipeline ,sentence):
  org_text = sentence
  sentence = clean(sentence)
  l = []
  if len(sentence.split()) < 15 and len(sentence.split()) > 2:
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    for n in range(1,4):
      for i in range(2):
        pred = generation_pipeline(sentence,
          return_full_text = False,
          pad_token_id=tokenizer.eos_token_id,
          num_beams=10 ,
          max_length=len(input_ids[0]) + n,
          top_p=0.9,
          repetition_penalty = 3.0,
          no_repeat_ngram_size = 3)[0]['generated_text'].replace("."," ").replace("،"," ").replace(":"," ").strip()
        pred = " ".join(pred.split()).strip()
        if not pred == "":
          pred = "*" + pred.replace(" ","_")
          aug = strip_punc(org_text) + " " + pred
          org_text = " ".join(org_text.split())
          pred = org_text.replace(strip_punc(org_text),aug)
          if not pred in l and not pred == org_text:
            l.append(pred)
  return l

# text here can be list of sentences or one string sentence
def aug_GPT(model_name,text):
  try:
    print("loading GPT... 🚀")
    tic = time.perf_counter()
    model , tokenizer , generation_pipeline = load_GPT(model_name)
    toc = time.perf_counter()
    print("loading GPT done ✅: " + str(toc-tic) + " seconds")
    print("augmenting with GPT... 🚀")
    tic = time.perf_counter()
    if isinstance(text, str):
      ret = GPT(model,tokenizer , generation_pipeline ,text)
      toc = time.perf_counter()
      print("augmenting with GPT done ✅: " + str(toc-tic) + " seconds")
      return ret
    else:
      all_sentences = []
      for sentence in text:
        sentence = sentence.strip()
        all_sentences.append([sentence,GPT(model,tokenizer , generation_pipeline ,sentence)])
      toc = time.perf_counter()
      print("augmenting with GPT done ✅: " + str(toc-tic) + " seconds")
      return all_sentences
  except:
    pass

# ------------------------- End of GPT2 Augmentation ---------------------------- #

# ------------------------------ W2V Augmentation ---------------------------- #

def load_w2v(ar_model,en_model):
  try:
      ar_model = gensim.models.KeyedVectors.load_word2vec_format(ar_model,binary=True,unicode_errors='ignore')
  except:
      ar_model = gensim.models.Word2Vec.load(ar_model)
  en_model = gensim.downloader.load(en_model)
  return ar_model, en_model

def w2v(ar_model,en_model,sentence):
  cleaned = clean(sentence)
  sentence = seperate_punct(sentence)
  l = []
  augs = []
  if len(sentence.split()) < 20 and len(sentence.split()) > 2:
    for i,token in enumerate(sentence.split()):
      pos_dict = pos(cleaned)
      if token in cleaned and is_replacable(token,pos_dict):
        if pos_dict[token] == 'FOREIGN':
            model_to_use = en_model
        else:
            model_to_use = ar_model
        try:
          word_vectors = model_to_use.wv
          if token in word_vectors.key_to_index:
            exist = True
          else:
            exist = False
        except:
          if token in model_to_use:
            exist = True
          else:
            exist = False
        if exist:
          try:
            most_similar = model_to_use.wv.most_similar( token, topn=5 )
          except:
            most_similar = model_to_use.most_similar( token, topn=5 )
          for term, score in most_similar:
                if term != token:
                    term = "*" + term
                    s = sentence.split()
                    s[i] = term
                    aug = " ".join(s)
                    if not clean(aug) in augs:
                      augs.append(clean(aug))
                      aug = " ".join(aug.split())
                      l.append(aug)
  return l

# text here is a list of sentences or one string sentence
def aug_w2v(ar_model,en_model,text):
  try:
    print("loading w2v... 🚀")
    tic = time.perf_counter()
    ar_model,en_model = load_w2v(ar_model,en_model)
    toc = time.perf_counter()
    print("loading w2v done ✅: " + str(toc-tic) + " seconds")
    print("augmenting with w2v... 🚀")
    tic = time.perf_counter()
    if isinstance(text, str):
      ret = w2v(ar_model,en_model,text)
      toc = time.perf_counter()
      print("augmenting with w2v done ✅: " + str(toc-tic) + " seconds")
      return ret
    else:
      all_sentences = []
      for sentence in text:
        sentence = sentence.strip()
        all_sentences.append([sentence,w2v(ar_model,en_model,sentence)])
      toc = time.perf_counter()
      print("augmenting with w2v done ✅: " + str(toc-tic) + " seconds")
      return all_sentences
  except:
    pass


# ------------------------- End of W2V Augmentation ---------------------------- #

# ------------------------------ BERT Augmentation ---------------------------- #

def load_bert(model):
  model = pipeline('fill-mask', model= model)
  return model

# Contextual word embeddings
def bert(model, sentence):
  cleaned = clean(sentence)
  sentence = seperate_punct(sentence)
  l = []
  augs = []
  if len(sentence.split()) < 15 and len(sentence.split()) > 2:
    for n,token in enumerate(sentence.split()):
        if token in cleaned and is_replacable(token,pos(sentence)):
          s = sentence.split()
          try:
            s[n] = "<mask>"
            masked_text = " ".join(s)
            pred = model(masked_text , top_k = 5)
          except:
            s[n] = "[MASK]"
            masked_text = " ".join(s)
            pred = model(masked_text , top_k = 5)
          for i in pred:
            if isinstance(i, dict):
              output = i['token_str']
              if not output == token:
                if not len(output) < 2 and clean(output) == output:
                  term = "*"+i['token_str']
                  s = sentence.split()
                  s[n] = term
                  aug = " ".join(s)
                  if not clean(aug) in augs:
                        augs.append(clean(aug))
                        aug = " ".join(aug.split())
                        l.append(aug)
  return l

def multi_bert(model, sentence):
    l = bert(model, sentence)
    ret = []
    for i in l:
      ret += bert(model, i)
    return ret

# text here is a list of sentences or one string sentence
def aug_bert(model, text, model_name):
  try:
    print(f"loading {model_name}... 🚀")
    tic = time.perf_counter()
    model = load_bert(model)
    toc = time.perf_counter()
    print(f"loading {model_name} done ✅: " + str(toc-tic) + " seconds")
    print(f"augmenting with {model_name}... 🚀")
    tic = time.perf_counter()
    if isinstance(text, str):
      ret = multi_bert(model, text)
      toc = time.perf_counter()
      print(f"augmenting with {model_name} done ✅: " + str(toc-tic) + " seconds")
      return ret
    else:
      all_sentences = []
      for sentence in text:
        sentence = sentence.strip()
        all_sentences.append([sentence,multi_bert(model,sentence)])
      toc = time.perf_counter()
      print(f"augmenting with {model_name} done ✅: " + str(toc-tic) + " seconds")
      return all_sentences
  except:
    pass

# -------------------------- End of BERT Augmentation ------------------------- #

# -------------------------- Back Translation Augmentation ------------------------- #

def load_models_bt(from_model_name, to_model_name):
  device = 'cpu'
  if tf.test.gpu_device_name():
      device = 'cuda'
  back_translation = naw.BackTranslationAug(
      from_model_name=from_model_name,
      to_model_name=to_model_name,
      device=device,
  )
  return (back_translation)

def aug_back_translate(text):
  try:
    all_sentences = []
    available_languages = ['ar-en', 'ar-fr', 'ar-tr', 'ar-ru',
                            'ar-pl', 'ar-it', 'ar-es', 'ar-de', 'ar-he']

    print("Loading and Augmenting Back Translating Models... 🚀")
    tic = time.perf_counter()

    for model in available_languages:
        model_name = model.split('-')
        back_translation = load_models_bt(f'Helsinki-NLP/opus-mt-{model_name[0]}-{model_name[1]}',
                                          'UBC-NLP/turjuman')
        bt_sentence_1 = back_translation.augment(text)
        all_sentences.append(bt_sentence_1)

    toc = time.perf_counter()
    print("Loading and Augmenting Back Translating Models done ✅: " + str(round(toc-tic, 3)) + "seconds")

    return all_sentences
  except:
    pass

# -------------------------- End of Back Translation Augmentation ---------------- #

# -------------------------- Similarity Checker --------------------------------- #

def load_similarity_checker_model(model_name):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModel.from_pretrained(model_name)
  return tokenizer, model

def similarity_checker(sentences, user_text_input, model_name):
  tokenizer, model = load_similarity_checker_model(
      'sentence-transformers/bert-base-nli-mean-tokens')
  average_similarity = 0

  try:
      if (len(sentences) > 0) and (not any(isinstance(sent, type(None)) for sent in sentences)):
          tokens = {'input_ids': [], 'attention_mask': []}
          sentences.insert(0, user_text_input)
          for sentence in sentences:
              # tokenize sentence and append to dictionary lists
              new_tokens = tokenizer.encode_plus(sentence, max_length=128, truncation=True,
                                                  padding='max_length', return_tensors='pt')
              tokens['input_ids'].append(new_tokens['input_ids'][0])
              tokens['attention_mask'].append(
                  new_tokens['attention_mask'][0])

          # reformat list of tensors into single tensor
          tokens['input_ids'] = torch.stack(tokens['input_ids'])
          tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

          outputs = model(**tokens)
          embeddings = outputs.last_hidden_state
          attention_mask = tokens['attention_mask']
          mask = attention_mask.unsqueeze(
              -1).expand(embeddings.size()).float()

          masked_embeddings = embeddings * mask
          summed = torch.sum(masked_embeddings, 1)
          summed_mask = torch.clamp(mask.sum(1), min=1e-9)
          mean_pooled = summed / summed_mask

          # Convert from PyTorch tensor to numpy array
          mean_pooled = mean_pooled.detach().numpy()

          # Calculate cosine similarity
          cos_similarity = cosine_similarity(
              [mean_pooled[0]], mean_pooled[1:])

          # Calculate average of similarities
          if len(sentences) >= 2:
              try:
                  average_similarity = mean(cos_similarity[0][1:])
              except:
                  print(
                      "No augmented sentences by the model. So average similarity was not calculated."
                      )

          return np.around(cos_similarity[0], decimals=6), average_similarity
  except:
      print(f"No augmented sentences by {model_name}.")
      pass

def similarity_table(sentences_list, similarity_list, model_name):
  model_name_list = []
  back_translation_languages = ['Back Translation (EN)', 'Back Translation (FR)', 'Back Translation (TR)',
                                'Back Translation (RU)', 'Back Translation (PL)', 'Back Translation (IT)',
                                'Back Translation (ES)', 'Back Translation (DE)', 'Back Translation (HE)']
  bt_counter = 0
  try:
    if (len(sentences_list) > 0) and (not any(isinstance(sent, type(None)) for sent in sentences_list)):
      for i in range(len(sentences_list)):
        if model_name == "Back Translation":
            model_name_list.append(back_translation_languages[bt_counter])
            bt_counter += 1
        else:
            model_name_list.append(model_name)

      data = list(zip(sentences_list, similarity_list, model_name_list))
      df = pd.DataFrame(data, columns=['Sentences', 'Similarity Score', 'Model Name'])
      return df
  except:
    print(f"No augmented sentences by {model_name}.")
    pass

# -------------------------- End of Similarity Checker ------------------------- #