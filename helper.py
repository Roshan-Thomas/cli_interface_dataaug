## Imports
import os
import json
import requests
import requests
from camel_tools.utils.charsets import AR_LETTERS_CHARSET
import string
import pandas as pd
import time
from models import (
    aug_GPT,aug_w2v, aug_bert,
    aug_back_translate, similarity_checker, similarity_table
)


def is_replacable(token,pos_dict):
  #  if ner(token) != 'O':
  #    return False
   if token in pos_dict:
    if bool(set(pos_dict[token].split("+")) & set(['NOUN','V','ADJ','FOREIGN'])):
      return True
   return False

def ner(text):
  url = 'https://farasa.qcri.org/webapi/ner/'
  api_key = "KMxvdPGsKHXQAbRXGL"
  payload = {'text': text, 'api_key': api_key}
  data = requests.post(url, data=payload)
  result = json.loads(data.text)
  return result['text'][0].split("/")[1]

def pos(text):
  url = 'https://farasa.qcri.org/webapi/pos/'
  api_key = "KMxvdPGsKHXQAbRXGL"
  payload = {'text': text, 'api_key': api_key}
  data = requests.post(url, data=payload)
  result = json.loads(data.text)
  text  = text.split()
  pos_dict  = {}
  for n in range(len(result["text"])):
    i = result["text"][n]
    if "+" == i['surface'][0]:
      word = "".join(s.strip() for s in result["text"][n-1]['surface'].split("+"))
      word = word + i['surface'].replace("+","").strip()
      if word in text:
        pos_dict[word] = result["text"][n-1]['POS']
    if "+" == i['surface'][-1]:
      word = "".join(s.strip() for s in result["text"][n+1]['surface'].split("+"))
      word = i['surface'].replace("+","").strip() + word
      if word in text:
       pos_dict[word] = result["text"][n+1]['POS']
    else:
      word = "".join(s.strip() for s in i['surface'].split("+"))
      if word in text:
        pos_dict[word] = i['POS']
  return pos_dict

def seperate_punct(text):
  text = text.strip()
  text = " ".join(text.split())
  ret = ""
  letter = str("".join(AR_LETTERS_CHARSET) + string.ascii_letters )
  for i,l in enumerate(text):
    if not i == len(text) - 1:
      if (l in letter or l == "*") and text[i+1] != " " and not text[i+1] in letter:
        ret += l + " "
      elif not (l in letter or l == "*") and text[i+1] != " " and text[i+1] in letter:
        ret += l + " "
      else:
        ret += l
    else:
      ret += l
  ret = " ".join(ret.split())
  return ret

def clean(text):
  # remove any punctuations in the text
  punc = """،.:!?؟!:.,''!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"""
  for l in text:
    if l in punc and l != " ":
      text = text.replace(l,"")
  # keep only arabic text
  # text = " ".join(re.findall(r'[\u0600-\u06FF]+', text))
  return text

def strip_punc(text):
  remove = ""
  for l in reversed(text):
    if l in AR_LETTERS_CHARSET:
      break
    elif not l in AR_LETTERS_CHARSET:
      remove += l
  return text.replace(remove[::-1],"")

def convert_df_to_csv(df, sentence_id, run_number, madar_iwslt):
    directory = f"run_{run_number}"
    try:
        os.makedirs(f"./runs/{directory}", exist_ok=True)
        print("Directory '%s' created successfully" % directory)
    except OSError as error:
        print(f"Directory '{directory}' cannot be created")
    
    if madar_iwslt == "MADAR":
        return df.to_csv(f'./runs/{directory}/madar/{sentence_id}.csv', index=False, encoding="utf-8-sig")
    elif madar_iwslt == "IWSLT":
        return df.to_csv(f'./runs/{directory}/iwslt/{sentence_id}.csv', index=False, encoding="utf-8-sig")

def execution(sentence):
  print("Beginning Augmentation... 🚀\n")

  # Augment sentences by each model
  arabert_sentences = aug_bert("aubmindlab/bert-large-arabertv2", sentence,"Arabert")
  qarib_bert_sentences = aug_bert("qarib/bert-base-qarib", sentence, "Qarib Bert")
  xlm_roberta_bert_sentences = aug_bert("xlm-roberta-base", sentence, "XLM-Roberta")
  arabart_sentences = aug_bert("moussaKam/AraBART", sentence, "Arabart")
  camel_bert_sentences = aug_bert("CAMeL-Lab/bert-base-arabic-camelbert-mix", sentence, "Camel Bert")
  bert_large_arabic_sentences = aug_bert("asafaya/bert-large-arabic", sentence, "Bert Large Arabic")
  arbert_sentences = aug_bert("UBC-NLP/ARBERT", sentence, "Arbert")
  marbert_sentences = aug_bert("UBC-NLP/MARBERTv2", sentence, "Marbert")
  araelectra_sentences = aug_bert("aubmindlab/araelectra-base-generator", sentence, "Araelectra")
  aragpt2_sentences = aug_GPT("aubmindlab/aragpt2-medium", sentence)
  aravec_sentences = aug_w2v("full_grams_cbow_100_twitter.mdl", 'glove-twitter-25', sentence)
  back_translation_sentences = aug_back_translate(sentence)

  # Process to convert sentences to normal list (rather than a list of lists)
  back_translation_sentences_list = []
  for i in range(len(back_translation_sentences)):
    back_translation_sentences_list.append(back_translation_sentences[i][0])

  print("Augmentation completed ✅\n\n")

  print("Calculating Similarity Scores... 🚀\n")

  # Calculate similarity scores
  try:
    arabert_similiarty_list, arabert_average_similarity = similarity_checker(
        arabert_sentences, sentence, "Arabert"
    )
  except:
    print("No augmented sentences by Arabert")
    pass
  try:
    qarib_bert_similarity_list, qarib_bert_average_similarity = similarity_checker(
        qarib_bert_sentences, sentence, "Qarib Bert"
    )
  except:
    print("No augmented sentences by Qarib Bert")
    pass
  try:
    xlm_roberta_similarity_list, xlm_roberta_average_similarity = similarity_checker(
        xlm_roberta_bert_sentences, sentence, "XLM Roberta"
    )
  except:
    print("No augmented sentences by XLM Roberta")
    pass
  try:
    arabart_similarity_list, arabart_average_similarity = similarity_checker(
        arabart_sentences, sentence, "Arabart"
    )
  except:
    print("No augmented sentences by Arabart")
    pass
  try:
    camel_bert_similarity_list, camel_bert_average_similarity = similarity_checker(
        camel_bert_sentences, sentence, "Camel Bert"
    )
  except:
    print("No augmented sentences by Camel Bert")
    pass
  try:
    bert_large_arabic_similarity_list, bert_large_arabic_average_similarity = similarity_checker(
        bert_large_arabic_sentences, sentence, "Bert Large Arabic"
    )
  except:
    print("No augmented sentences by Bert Large Arabic")
    pass
  try:
    arbert_similarity_list, arbert_average_similarity = similarity_checker(
        arbert_sentences, sentence, "Arbert"
    )
  except:
    print("No augmented sentences by Arbert")
    pass
  try:
    marbert_similarity_list, marbert_average_similarity = similarity_checker(
        marbert_sentences, sentence, "Marbert"
    )
  except:
    print("No augmented sentences by Marbert")
    pass
  try:
    araelectra_similarity_list, araelectra_average_similarity = similarity_checker(
        araelectra_sentences, sentence, "Araelectra"
    )
  except:
    print("No augmented sentences by Araelectra")
    pass
  try:
    aragpt2_similarity_list, aragpt2_average_similarity = similarity_checker(
        aragpt2_sentences, sentence, "AraGPT2"
    )
  except:
    print("No augmented sentences by AraGPT2")
    pass
  try:
    aravec_similarity_list, aravec_average_similarity = similarity_checker(
        aravec_sentences, sentence, "Aravec"
    )
  except:
    print("No augmented sentences by Aravec (w2v)")
    pass
  try:
    back_translated_similarity_list, back_translated_average_similarity = similarity_checker(
        back_translation_sentences_list, sentence, "Back Translation"
    )
  except:
    print("No augmented sentences by Back Translation")
    pass

  print("Similarity Lists completed ✅\n\n")

  print("Creating Dataframes... 🚀\n")

  # Dataframes Tables (Similarity and Sentences)
  arabert_df = qarib_bert_df = xlm_roberta_bert_df = arabart_df = camel_bert_df = bert_large_arabic_df = []
  arbert_df = marbert_df = araelectra_df = aragpt2_df = aravec_df = back_translated_df = []

  try:
    arabert_df = similarity_table(arabert_sentences, arabert_similiarty_list, "Arabert")
  except:
    pass
  try:
    qarib_bert_df = similarity_table(qarib_bert_sentences, qarib_bert_similarity_list, "Qarib")
  except:
    pass
  try:
    xlm_roberta_bert_df = similarity_table(xlm_roberta_bert_sentences, xlm_roberta_similarity_list, "XLM-Roberta")
  except:
    pass
  try:
    arabart_df = similarity_table(arabart_sentences, arabart_similarity_list, "Arabart")
  except:
    pass
  try:
    camel_bert_df = similarity_table(camel_bert_sentences, camel_bert_similarity_list, "Camel Bert")
  except:
    pass
  try:
    bert_large_arabic_df = similarity_table(bert_large_arabic_sentences, bert_large_arabic_similarity_list, "Bert Large Arabic")
  except:
    pass
  try:
    arbert_df = similarity_table(arbert_sentences, arbert_similarity_list, "Arbert")
  except:
    pass
  try:
    marbert_df = similarity_table(marbert_sentences, marbert_similarity_list, "Marbert")
  except:
    pass
  try:
    araelectra_df = similarity_table(araelectra_sentences, araelectra_similarity_list, "Araelectra")
  except:
    pass
  try:
    aragpt2_df = similarity_table(aragpt2_sentences, aragpt2_similarity_list, "AraGPT2")
  except:
    pass
  try:
    aravec_df = similarity_table(aravec_sentences, aravec_similarity_list, "Aravec")
  except:
    pass
  try:
    back_translated_df = similarity_table(back_translation_sentences_list, back_translated_similarity_list, "Back Translation")
  except:
    pass

  print("Dataframes completed ✅\n\n")

  df_list = []

  all_df_list = [arabert_df, qarib_bert_df, xlm_roberta_bert_df, 
                      arabert_df, camel_bert_df, bert_large_arabic_df, arbert_df,
                      marbert_df, araelectra_df, aragpt2_df, aravec_df, 
                      back_translated_df] 
  
  for i in range(len(all_df_list)):
    if len(all_df_list[i]) > 0:
      df_list.append(all_df_list[i])

  all_df = pd.concat(df_list, ignore_index=True)

  toc = time.perf_counter()

  return all_df
