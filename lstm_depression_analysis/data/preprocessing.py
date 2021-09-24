
import re
import spacy
import nltk
import time

stop_words = ["vc", ",", ".", ":", "!", "vcs", "tô", "to", "n", "tbm", "tmb", "tá",
             "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é",
             "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as",
             "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à",
             "seu", "sua", "ou", "ser", "quando", "muito", "há", "nos", "já",
             "está", "eu", "também", "só", "pelo", "pela", "até", "isso",
             "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter",
             "seus", "quem", "nas", "me", "esse", "eles", "estão", "você",
             "tinha", "foram", "essa", "num", "nem", "suas", "meu", "às",
             "minha", "têm", "numa", "pelos", "elas", "havia", "seja", "qual",
             "será", "nós", "tenho", "lhe", "deles", "essas", "esses", "pelas",
             "este", "fosse", "dele", "tu", "te", "vocês", "vos", "lhes",
             "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso", "nossa",
             "nossos", "nossas", "dela", "delas", "esta", "estes", "estas",
             "aquele", "aquela", "aqueles", "aquelas", "isto", "aquilo",
             "estou", "está", "estamos", "estão", "estive", "esteve",
             "estivemos", "estiveram", "estava", "estávamos", "estavam",
             "estivera", "estivéramos", "esteja", "estejamos", "estejam",
             "estivesse", "estivéssemos", "estivessem", "estiver",
             "estivermos", "estiverem", "hei", "há", "havemos", "hão", "houve",
             "houvemos", "houveram", "houvera", "houvéramos", "haja",
             "hajamos", "hajam", "houvesse", "houvéssemos", "houvessem",
             "houver", "houvermos", "houverem", "houverei", "houverá",
             "houveremos", "houverão", "houveria", "houveríamos", "houveriam",
             "sou", "somos", "são", "era", "éramos", "eram", "fui", "foi",
             "fomos", "foram", "fora", "fôramos", "seja", "sejamos", "sejam",
             "fosse", "fôssemos", "fossem", "for", "formos", "forem", "serei",
             "será", "seremos", "serão", "seria", "seríamos", "seriam",
             "tenho", "tem", "temos", "tém", "tinha", "tínhamos", "tinham",
             "tive", "teve", "tivemos", "tiveram", "tivera", "tivéramos",
             "tenha", "tenhamos", "tenham", "tivesse", "tivéssemos",
             "tivessem", "tiver", "tivermos", "tiverem", "terei", "terá",
             "teremos", "terão", "teria", "teríamos", "teriam"]

class TokenizerMergeVocab():

    def __init__(self):
        self._nlp = spacy.load("pt_core_news_sm", disable=["ner", "tagger", "parser"])
        self._stemmer = nltk.stem.RSLPStemmer()

    def tokenize(self, text, remove_hashtags=True, stemming=False):
        text = text.lower().replace("\n", "").replace("\r", "")
        doc = self._nlp(text)
        return self._normalize(doc, remove_hashtags, stemming)

    def _normalize(self, doc, remove_hashtags, stemming):
        tokens = []

        remove_next = False
        for token in doc:
            tk = token.text

            re_username = re.compile(r"^@\w+|\s@\w+", re.UNICODE)
            re_transform_url = re.compile(r'(http|https)://[^\s]+', re.UNICODE)
            re_transform_emails = re.compile(r'[^\s]+@[^\s]+', re.UNICODE)
            re_transform_numbers = re.compile(r'\d+', re.UNICODE)
            re_transform_retweet = re.compile(r'rt', re.UNICODE)

            re_transform_rr_three_times_or_more = re.compile(r'r{3,}', re.UNICODE)
            re_transform_ss_three_times_or_more = re.compile(r's{3,}', re.UNICODE)
            
            re_transform_laugh_haha = re.compile(r'(?:a*(?:ha)+h?)', re.UNICODE)
            re_transform_laugh_hehe = re.compile(r'(?:e*(?:he)+h?)', re.UNICODE)
            re_transform_laugh_hihi = re.compile(r'(?:i*(?:hi)+h?)', re.UNICODE)
            re_transform_laugh_hoho = re.compile(r'(?:o*(?:ho)+h?)', re.UNICODE)
            re_transform_laugh_huhu = re.compile(r'(?:u*(?:hu)+h?)', re.UNICODE)

            punctuations = re.escape(r'"_#°%\'()\*\+/=@\|{}~?')
            re_punctuations = re.compile(r'[%s]' % (punctuations), re.UNICODE)

            tk = re_username.sub("", tk)
            tk = re_transform_url.sub("", tk)
            tk = re_transform_emails.sub("", tk)
            tk = re_transform_retweet.sub("", tk)
            tk = re_transform_rr_three_times_or_more.sub("rr", tk)
            tk = re_transform_ss_three_times_or_more.sub("ss", tk)

            if remove_next:
                tk = ""
                remove_next = False

            if remove_hashtags and tk == "#":
                remove_next = True

            tk = re_transform_numbers.sub("0", tk)

            # Checking if token is composed by only 1 char
            if tk and self._is_token_made_by_only_one_char(tk):
                tk = tk[0]

            tk = re_punctuations.sub("", tk)
            
            # Laugh checking section
            laugh_token = "_laugh"
            if tk and re_transform_laugh_haha.findall(tk) and self._is_token_made_only_by_this_char(tk, ["h", "a"]):
                tk = laugh_token
                
            if tk and re_transform_laugh_hehe.findall(tk) and self._is_token_made_only_by_this_char(tk, ["h", "e"]):
                tk = laugh_token
                
            if tk and re_transform_laugh_hihi.findall(tk) and self._is_token_made_only_by_this_char(tk, ["h", "i"]):
                tk = laugh_token
                
            if tk and re_transform_laugh_hoho.findall(tk) and self._is_token_made_only_by_this_char(tk, ["h", "o"]):
                tk = laugh_token
                
            if tk and re_transform_laugh_huhu.findall(tk) and self._is_token_made_only_by_this_char(tk, ["h", "u"]):
                tk = laugh_token
                
            if tk and self._is_token_made_only_by_this_char(tk, ["k"]):
                tk = laugh_token
                
            if tk and self._is_token_made_only_by_this_char(tk, ["a", "s", "k", "o"]):
                tk = laugh_token
                
            if tk and self._is_token_made_only_by_this_char(tk, ["h", "a", "u"]):
                tk = laugh_token

            if tk and self._is_token_made_only_by_this_char(tk, ["h", "a", "u", "s"]):
                tk = laugh_token

            if tk and stemming:
                tk = self._stemmer.stem(tk)

            if tk and (" " not in tk) and tk not in stop_words:
                tokens.append(tk)
        return tokens

    def _is_token_made_only_by_this_char(self, token, char_list):
        token_chars = []        
        for i in range(0, len(token)) : 
            if token[i] not in char_list:                 
                return False
            else:
                if token[i] not in token_chars:
                    token_chars.append(token[i])

        if len(token_chars) == len(char_list):
            return True
        else:
            return False

    def _is_token_made_by_only_one_char(self, token):
        for i in range(1, len(token)) : 
            if token[i] != token[0] : 
                return False
        return True
