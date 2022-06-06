

import numpy as np
import pandas as pd
import math
import cv2
import nltk
from nltk.tokenize import word_tokenize


df=pd.read_excel(r"C:\Users\user\Documents\Django\Alumini\Alumini\Alumini\BigData.xlsx")

def replace_fn(x) :
  if(type(x) is not str):
    return x
  e = ['Ã©' , 'Ã¨' , 'Ã']
  for el in e  :
    x = x.replace(el,'é')
  x = x.replace('âÃ´','o')
  x = x.replace('Ã´','o')
  x = x.replace('Ã§','c')
  x = x.replace('â',"'")
  x = x.replace("Ã","î")
  return x

'''
df = df.drop(columns=['location_name','member_id','profile_url',
                      'headline','address','organization_start_1','organization_end_1',
                      'organization_location_1','organization_start_2',
                      'organization_end_2','organization_location_2',
                      'organization_start_3','organization_end_3','organization_location_3',
                      'organization_start_4','organization_end_4','organization_location_4',
                      'organization_start_5','organization_end_5','organization_location_5',
                      'organization_end_6'])
df = df.drop(columns=['full_name'])
'''
df = df.applymap(replace_fn)

def profile_fn(tokenized_doc):
  scores_global = []
  k = 0
  for i, person_elements in enumerate(df.iterrows()):
    k = k + 1
    for person_element in person_elements[1]:
      tokenized_per = []
      if (isinstance(person_element, str) and person_element != 'None'):
        tokenized_per.extend(word_tokenize(person_element.lower()))
    a_set = set(tokenized_per)
    b_set = set(tokenized_doc[0])
    nb = 0
    if a_set & b_set:
      nb = len(a_set & b_set)
    scores_global.append(nb / len(tokenized_doc[0]))
  indices = np.array(scores_global).argsort()[::-1][:len(scores_global)]
  return df.iloc[indices[0:3],:]
