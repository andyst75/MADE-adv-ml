from zipfile import ZipFile
import random
from collections import Counter
import numpy as np
from typing  import List
import re


ALPHABET = list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя ')
ENCODE_CHAR_SHIFT = 0
SEED = 1234

def read_file_from_archive(archive: ZipFile, name: str) -> str:
    return archive.read(name).decode("utf-8").replace('\t', ' ')


def text_to_str(text: str, alphabet: list = ALPHABET, min_len: int = 20) -> str:
    list_of_str = [x.lower() for x in text.split(sep='\n') 
                   if len(x) >= min_len]
    
    text_str = ' '.join(list_of_str)
    text_out = re.sub(f"[^{''.join(alphabet)}]+", "", text_str)\
        .replace("       ", " ")\
        .replace("      ", " ")\
        .replace("     ", " ")\
        .replace("    ", " ")\
        .replace("   ", " ")\
        .replace("  ", " ")\
        .replace("  ", " ")\
        .replace("  ", " ")
    return text_out.strip()


def get_ngrams(text: str, ngram: int) -> list:
    ngrams = [text[i : i + ngram] for i in range(len(text) - ngram + 1)]
    return ngrams


def get_ngram_counter(text: str, ngram: int = 1) -> Counter:
    ngrams = get_ngrams(text, max(1, ngram))
    return Counter(ngrams)


def calc_ngram_freq(text: str, ngram: int = 1) -> dict:
    ngrams = get_ngrams(text, max(1, ngram))
    freq_ngram = Counter(ngrams)
    total_ngrams = len(ngrams)
    freq = Counter({k: v / total_ngrams for k, v in freq_ngram.items()})
    return dict(freq)


def encode_char_map(alphabet: list = ALPHABET,
                    char_shift: int = ENCODE_CHAR_SHIFT,
                    seed: int = SEED) -> dict:
    np.random.seed(seed)
    cipher_alphabet = [chr(ord(x) + char_shift) for x in alphabet]
    np.random.shuffle(cipher_alphabet) 
    char_dict = dict(zip(alphabet, cipher_alphabet))
    return char_dict


def text_encoder(text: str, 
                 alphabet: list = ALPHABET,
                 char_shift: int = ENCODE_CHAR_SHIFT,
                 seed: int = SEED,
                 encode_dict: dict = None) -> str:

    if encode_dict is None:
        encode_dict = encode_char_map(alphabet, char_shift, seed)

    text_encoded = [encode_dict[c] for c in text_to_str(text, alphabet)]
    
    return ''.join(text_encoded)


def char_accuracy(pred_text: str, true_text: str) -> float:
    assert len(pred_text) == len(true_text)
    true = sum([pred_text[i] == true_text[i] for i in range(len(pred_text))])
    return true / len(pred_text)


def apply_ngram(ngram_dict: dict, enc_dict: dict, enc_text: str, ngram: int=2) -> str:
    
    enc_freq = calc_ngram_freq(enc_text, ngram=ngram)
    
    res = list(enc_text)
    for key, _ in sorted(enc_freq.items(), key=lambda x: x[1]):
        pos = [m.start() for m in re.finditer(key, enc_text)]
        for i in pos:
            res[i : i+ngram] = ngram_dict[key]
    return "".join(res)


class MCMCModel:
    def __init__(self,
                 text_corpus: str, 
                 n_gram: int = 2,
                 n_iter: int = 10_000,
                 seed: int=SEED):

        np.random.seed(seed)
        self.unifreqs = calc_ngram_freq(text_corpus, 1)
        self.freqs = calc_ngram_freq(text_corpus, n_gram) if n_gram > 1 else self.unifreqs
        self.n_gram = n_gram
        self.n_iter = n_iter     

    def _decode(self, text: str, decode_dict: dict) -> str:
        return ''.join(decode_dict[c] for c in text if c in decode_dict)
    
    def _log_likelihood(self, text: str) -> float:
        ngram_counter = get_ngram_counter(text, self.n_gram)
        none_freq = min(self.freqs.values()) / 2
        result = 0
        for ngram, count in ngram_counter.most_common():
            freq = self.freqs.get(ngram)
            if freq is None:
                freq = none_freq
                none_freq /= 2 
            result += count * np.log(freq)
        return result

    def _random_permute_dict(self, decode_dict: dict):
        perm_decode_dict = decode_dict.copy()
        k1, k2 = np.random.choice(list(perm_decode_dict.keys()), 2, replace=False)
        perm_decode_dict[k1], perm_decode_dict[k2] = \
            perm_decode_dict[k2], perm_decode_dict[k1]
        return perm_decode_dict
   
    def fit(self, encoded_text: str):
        corpus_freq = [x[0] for x in sorted(self.unifreqs.items(),
                                            key=lambda x: x[1], reverse=True)]
        encoded_freq = [x[0] for x in sorted(calc_ngram_freq(encoded_text, 1).items(),
                                             key=lambda x: x[1], reverse=True)]

        decode_dict = dict(zip(encoded_freq, corpus_freq))
        best_decode_dict = decode_dict
        
        max_llh = self._log_likelihood(self._decode(encoded_text, decode_dict))
        cur_llh = max_llh
        for i in range(self.n_iter):
            new_decode_dict = self._random_permute_dict(decode_dict)
            new_text = self._decode(encoded_text, new_decode_dict)
            new_llh = self._log_likelihood(new_text)
            if new_llh > cur_llh or np.random.rand() < np.exp(new_llh - cur_llh):
                decode_dict = new_decode_dict
                cur_llh = new_llh
                if cur_llh > max_llh:
                    max_llh = cur_llh
                    best_decode_dict = decode_dict
        self.decode_dict = best_decode_dict
        return self
    
    def predict(self, encoded_text: str) -> str:
        return self._decode(encoded_text, self.decode_dict)
