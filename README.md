# Natural-Language-Processing-Skills

**Natural Language Processing using Machine Leaning:**

**My Skills in NLP:**

**1. Tokenization**

**2. Stemming**

**3. Lemmatization**

**4. Stop Words**

1. **Tokenization** - Corpus, Document, Vocabulary, Words



**NLP Libraries: NLTK, spaCy**


**(i)** **NLTK** - ntlk.sent_tokenize, nltk.word_tokenize

**(ii) Tokenization using NLTK: ** from nltk.tokenize import wordpunct_tokenize (Splits for Punctuations too), from nltk.tokenize import TreebankWordTokenizer (Full Stop won't be considered as a word in in-between places)

**(iii) Stemming: ** from nltk.stem import PorterStemmer

**Issue:** (With PorterStummer) Some words loose it's form E.g. history---->histori; congratulations --> congratuli

B) from nltk.stem import RegexpStemmer;   reg_stemmer=RegexpStemmer('ing$|s$|e$|able$', min=4) [eating --> eat)

C) from nltk.stem import SnowballStemmer; Performs better than PorterStemmer;  from nltk.stem import SnowballStemmer; 

Example: stemming.stem("fairly"),stemming.stem("sportingly") --> ('fairli', 'sportingli')

snowballsstemmer.stem("fairly"),snowballsstemmer.stem("sportingly") --> ('fair', 'sport')

**Snowball Stemmer better than Porter Stemmer**

**(iv) Lemmatization: ** Stemming - Word Stems; Lemmatization --> Lemma (Root Words rather than Word Stems) (Done using WordNetCorpusReader); lemmatizer.lemmatize("going",pos='v') [Better for Verbs]

**NLTK provides WordNetLemmatizer class which is a thin wrapper around the wordnet corpus. This takes more time because it gets word from the Corpus**




