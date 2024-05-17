import re
import spacy
from typing import List, Dict, Any, Set, Tuple
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch
from torch import nn
from transformers import set_seed

tokenizer_pattern: str = r"\b\w[\w']*\b"

stemmer = PorterStemmer()

def clean_text(text):
    # Menghapus @mentions
    text = re.sub(r'@\w+', '', text)

    # Menghapus URL
    text = re.sub(r'http\S+|www.\S+', '', text)

    # Menghapus emoticon
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)

    # Menghapus karakter non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Menghapus karakter tertentu
    text = re.sub(r'[\^*\\~-].*', '', text)

    # Menghapus new line
    text = re.sub(r'\n', ' ', text)

    return text

def tokenize_text_en(text: str, tokenizer_pattern: str) -> List[str]:
  tokens: List[str] = re.findall(tokenizer_pattern, text)
  return tokens
# Lemmatize
nlp = spacy.load("en_core_web_sm")

def lemmatize_tokens_en(tokens: List[str], nlp) -> List[str]:
  example_lemmatized_en = []

  for doc in nlp.pipe(tokens):
    tok = [token.lemma_ for token in doc]
    example_lemmatized_en.extend(tok)

  return example_lemmatized_en

def stem_tokens_en(tokens: List[str], stemmer: PorterStemmer) -> List[str]:
  stemmed_tokens = [stemmer.stem(token) for token in tokens]
  return stemmed_tokens

nltk_stop_words_list: List[str] = stopwords.words('english')
nltk_stop_words_set: Set[str] = set(nltk_stop_words_list)

def remove_stop_words_en(tokens: List[str], stop_words: Dict[str, Any]) -> List[str]:
  tokens_without_stop_words: List[str] = [
      token
      for token in tokens
      if token not in stop_words
  ]
  return tokens_without_stop_words

def join_words_en(tokens: List[str]) -> str:
  words: str = ' '.join(tokens)
  return words

def preprocess_text_en(text: str,
                                   stemmer: PorterStemmer,
                                   tokenizer_pattern: str,
                                   nlp) -> str:
  tokens: List[str] = tokenize_text_en(
    text = text,
    tokenizer_pattern = tokenizer_pattern,
  )
  tokens: List[str] = lemmatize_tokens_en(
    tokens = tokens,
    nlp = nlp,
  )
  tokens: List[str] = stem_tokens_en(
    tokens = tokens,
    stemmer = stemmer,
  )
  tokens: List[str] = remove_stop_words_en(
    tokens = tokens,
    stop_words = nltk_stop_words_set,
  )
  words: str = join_words_en(
      tokens = tokens,
  )
  return words

def clean_answer(text):
  split_output = text.split('### Answer:')
  answer_split = split_output[1].split(' ')
  answer = ''
  index=0
  for i in answer_split:
      if i!='':
          index+=1
      if index > 2 and i=='':
          break
      else:
          answer = answer+i+' '
  answer = answer.split()
  answer = ' '.join(answer)
  return answer

set_seed(87)
def generate_text_sampling_top_p_nucleus_22(
    input_prompt: str,
    min_length: int = 3,
    max_length: int =256,
    top_p: float = 0.22,
  ) -> str:
  input_prompt: str = instruction_format.format(
      question=input_prompt,
      answer='',
  )
  encoded_input: BatchEncoding = tokenizer(input_prompt, return_tensors='pt').to(device)
  sampling_output_tensor: Tensor = model.generate(
      **encoded_input,
      min_length=min_length,
      max_length=max_length,
      pad_token_id=50256,
      do_sample=True,
      top_p=top_p,
      top_k=0,
  )
  sampling_output_text: str = tokenizer.batch_decode(sampling_output_tensor, skip_special_tokens=True)[0]
  answer = clean_answer(sampling_output_text)
  return answer

def preprocess_question(text):
    prep1 = clean_text(text)
    return preprocess_text_en(prep1, stemmer, tokenizer_pattern, nlp)

device: torch.device = torch.device("cuda") \
  if torch.cuda.is_available() else torch.device("cpu")

instruction_format: str = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n"
    "\n"
    "### Question:\n"
    "{question}"
    "\n\n"
    "### Answer:\n"
    "{answer}"
)
    
tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model: nn.Module = AutoModelForCausalLM.from_pretrained('../gpt2')