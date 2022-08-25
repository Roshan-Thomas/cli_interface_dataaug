## Imports
import pandas as pd 
import time
import logging
from helper import (
    convert_df_to_csv, execution
)


# ------------------------------- Main Execution Code Block ----------------------- #

run_number = 1  # Change this number every run

logging.basicConfig(filename=f'./logs/run_{run_number}.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)

# MADAR Sentences Augmentation
madar_df = pd.read_csv("MADAR.csv")
sentences_list_madar = madar_df.MSA
sentences_id_list_madar = madar_df.ID

for i in range(len(sentences_list_madar)):
  tic = time.perf_counter()
  print(f"Augmenting line {i+1}\n\n")
  df = execution(f"{sentences_list_madar[i]}")
  convert_df_to_csv(df, sentences_id_list_madar[i], run_number, "MADAR")
  toc = time.perf_counter()
  print(f"\n\nCompleted augmenting line {i+1} in {str(toc-tic)} seconds\n")

# IWSLT Sentences Augmentation
iwslt_df = pd.read_csv("IWSLT.csv")
sentences_list_iwslt = iwslt_df.Sentence
sentences_id_list_iwslt = iwslt_df.ID

for i in range(len(sentences_list_iwslt)):
  tic = time.perf_counter()
  print(f"Augmenting line {i+1}\n\n")
  df = execution(f"{sentences_list_iwslt[i]}")
  convert_df_to_csv(df, sentences_id_list_iwslt[i], run_number, "IWSLT")
  toc = time.perf_counter()
  print(f"\n\nCompleted augmenting line {i+1} in {str(toc-tic)} seconds\n")

# ---------------------------- END of Execution Code Block ----------------------- #






