import torch
import torch.nn as nn

import pickle

from transformers import AutoModel, AutoTokenizer

from lstm_depression_analysis import settings
from lstm_depression_analysis.data.preprocessing import TokenizerMergeVocab

class DepressionCorpusTwitter(torch.utils.data.Dataset):

  def __init__(self, observation_period, dataset, subset, batch_size):

    """
      Params:

      subset: Can take three possible values: (train, test, val)
      observation_period: number of days for the period
      dataset: Cam tale ten possibile values: 0 to 9
      
    """
    
    pt = "neuralmind/bert-base-portuguese-cased"
    self.tokenizer = AutoTokenizer.from_pretrained(pt)

    subset_to_index = {"train": 0, "val": 1, "test": 2}
    self.subset_idx = subset_to_index[subset]
    self._dataset = dataset
    self._ob_period = int(observation_period)

    self._batch_size = batch_size

    self._tokenizer = TokenizerMergeVocab()

    self._raw = self.load_twitter_stratified_data()
    self._raw = self._raw["data_" + str(self._ob_period)][self._dataset]
    self._raw = self._raw[self.subset_idx]

    # o _data tem a estrutura (texto, label, user_name, post_date)
    # mas no fim, só uso por enquanto o texto e o label
    self._temp_data = self._get_posts_list_from_users(self._raw)
    self._data = self._get_bert_tokenizer_data(self._temp_data)
    
    
    # TODO: perguntar pq no readorsee faz isso? tem algo ligado a gpu?
    # eu tomei erro nisso aqui, mas n entendi mto bem pq ele tinha sobrado só 1 no batch msm e tal
    self._data = self.slice_if_rest_one(self._data, self._batch_size)

  def _get_max_length_tokenizer(self, users_data):
    max_length_tokenizer = 0
    for user_data in users_data:
      posts = user_data[0]
      for post in posts:
        post_len = len(post.split()) + 2
        if (post_len > max_length_tokenizer):
          max_length_tokenizer = post_len

    return max_length_tokenizer

  def _get_bert_tokenizer_data(self, data):
    data_to_return = []
    for user_data in data:
      posts = user_data[0]
      label = user_data[1]

      max_length_tokenizer = self._get_max_length_tokenizer(data)
  
      captions_tensors = self.tokenizer.batch_encode_plus(
          posts,
          add_special_tokens=True,
          # TODO: Fazer uma média, medianana e tb um histograma do dataset real para ficar dinâmico o valor do max_lengh :)
          max_length=256, # BERT: max is 512
          padding="max_length",
          return_tensors="pt",
          return_attention_mask=True,
          truncation=True,
      )

      # Pegando só os valores de input_ids e attention_mask
      captions_tensors = dict((k,captions_tensors[k]) for k in ('input_ids','attention_mask') if k in captions_tensors)
      
      data_to_return.append((captions_tensors, label))
        
    return data_to_return
    
    
  def load_twitter_stratified_data(self):
    
    twitter_stratified_path = settings.PATH_TO_TWITTER_STRATIFIED_DATA

    data = None
    with open(twitter_stratified_path, "rb") as f:
        data = pickle.load(f)
    return data

  def _get_posts_list_from_users(self, user_list):
    """ Return a list of posts from a user_list (all posts from each user)
    
    This function consider an instagram post with multiples images as 
    multiples posts with the same caption for all images in the same post.
    """
    data = []
    
    for u in user_list:
      u_name = u.username
      label = u.questionnaire.get_binary_bdi()

      u_posts = []

      for post in u.get_posts_from_qtnre_answer_date(self._ob_period):
        
        text = post.tweet_text
        date = post.date
        txt = self.preprocess_data(text)
        
        # Juntando o array de tokens de volta pra string
        text_tokenized = ' '.join(txt)

        u_posts.append(text_tokenized)

      data.append((u_posts, label, u_name, date))

    return data

  def preprocess_data(self, text):
    text = self._tokenizer.tokenize(text)[:100]
    text = [""] if not text else text

    return text

  def slice_if_rest_one(self, data, batch_size):
    data_size = len(data)
    num_gpus = 1
    #batch_size = batch_size / len(self.config.general["gpus"])
    
    # TODO: no nosso caso, to deixando como se tivesse só uma GPU
    batch_size = batch_size / num_gpus

    if data_size % batch_size == 1:
        return data[:-1]
    return data
    
  def __len__(self):
    return len(self._data)  # required

  def __getitem__(self, idx):
    captions_tensors, label = self._data[idx]
    
    return (captions_tensors, label)


class DepressionCorpusInsta(torch.utils.data.Dataset):

  def __init__(self, observation_period, dataset, subset, batch_size):

    """
      Params:

      subset: Can take three possible values: (train, test, val)
      observation_period: number of days for the period
      dataset: Cam tale ten possibile values: 0 to 9
      
    """
    
    pt = "neuralmind/bert-base-portuguese-cased"
    self.tokenizer = AutoTokenizer.from_pretrained(pt)

    subset_to_index = {"train": 0, "val": 1, "test": 2}
    self.subset_idx = subset_to_index[subset]
    self._dataset = dataset
    self._ob_period = int(observation_period)

    self._batch_size = batch_size

    self._tokenizer = TokenizerMergeVocab()

    self._raw = self.load_insta_stratified_data()
    self._raw = self._raw["data_" + str(self._ob_period)][self._dataset]
    self._raw = self._raw[self.subset_idx]

    # o _data tem a estrutura (texto, label, user_name, post_date)
    # mas no fim, só uso por enquanto o texto e o label
    self._temp_data = self._get_posts_list_from_users(self._raw)
    self._data = self._get_bert_tokenizer_data(self._temp_data)
    
    
    # TODO: perguntar pq no readorsee faz isso? tem algo ligado a gpu?
    # eu tomei erro nisso aqui, mas n entendi mto bem pq ele tinha sobrado só 1 no batch msm e tal
    self._data = self.slice_if_rest_one(self._data, self._batch_size)

  def _get_max_length_tokenizer(self, users_data):
    max_length_tokenizer = 0
    for user_data in users_data:
      posts = user_data[0]
      for post in posts:
        post_len = len(post.split()) + 2
        if (post_len > max_length_tokenizer):
          max_length_tokenizer = post_len

    return max_length_tokenizer

  def _get_bert_tokenizer_data(self, data):
    data_to_return = []
    for user_data in data:
      posts = user_data[0]
      label = user_data[1]

      max_length_tokenizer = self._get_max_length_tokenizer(data)
  
      captions_tensors = self.tokenizer.batch_encode_plus(
          posts,
          add_special_tokens=True,
          # TODO: Fazer uma média, medianana e tb um histograma do dataset real para ficar dinâmico o valor do max_lengh :)
          max_length=256, # BERT: max is 512
          padding="max_length",
          return_tensors="pt",
          return_attention_mask=True,
          truncation=True,
      )

      # Pegando só os valores de input_ids e attention_mask
      captions_tensors = dict((k,captions_tensors[k]) for k in ('input_ids','attention_mask') if k in captions_tensors)

      data_to_return.append((captions_tensors, label))
        
    return data_to_return
    
    
  def load_insta_stratified_data(self):
    
    insta_stratified_path = settings.PATH_TO_INSTA_STRATIFIED_DATA

    data = None
    with open(insta_stratified_path, "rb") as f:
        data = pickle.load(f)
    return data

  def _get_posts_list_from_users(self, user_list):
    """ Return a list of posts from a user_list (all posts from each user)
    
    This function consider an instagram post with multiples images as 
    multiples posts with the same caption for all images in the same post.
    """
    data = []
    
    for u in user_list:
      u_name = u.username
      label = u.questionnaire.get_binary_bdi()

      u_posts = []

      for post in u.get_posts_from_qtnre_answer_date(self._ob_period):
        
        text = post.caption
        date = post.date
        txt = self.preprocess_data(text)
        
        # Juntando o array de tokens de volta pra string
        text_tokenized = ' '.join(txt)

        u_posts.append(text_tokenized)
      
      data.append((u_posts, label, u_name, date))

    return data

  def preprocess_data(self, text):
    text = self._tokenizer.tokenize(text)[:100]
    text = [""] if not text else text

    return text

  def slice_if_rest_one(self, data, batch_size):
    data_size = len(data)
    num_gpus = 1
    #batch_size = batch_size / len(self.config.general["gpus"])
    
    # TODO: no nosso caso, to deixando como se tivesse só uma GPU
    batch_size = batch_size / num_gpus

    if data_size % batch_size == 1:
        return data[:-1]
    return data
    
  def __len__(self):
    return len(self._data)  # required

  def __getitem__(self, idx):
    captions_tensors, label = self._data[idx]
    
    return (captions_tensors, label)