# Natural-Language-Processing-Skills

**Natural Language Processing using Machine Leaning:**

**NLP Libraries: NLTK, spaCy**

**My Skills in NLP:**

**1. Tokenization**

**2. Stemming**

**3. Lemmatization**

**4. Stop Words**

**5. Parts of Speech Tagging**

**6. Named Entity Recognition**

**7. One-Hot Encoding**

**8. Bag of Words**

**NLP Projects Workflow:**

(i) Dataset Loading

(ii) Text Pre-Processing - I --> Tokenization, Lowercase the words, Regular Expressions

(iii) Text Pre-Processing - II --> Stemming, Lemmatization, Stop Words

(iv) Words to Vectors --> One-Hot Encoding, Bag of Words, TF-IDF, Word2vec, Avg Word2Vec

(v) ML Model Development


1. **Tokenization** - Corpus, Document, Vocabulary, Words

**(i)** **NLTK** - ntlk.sent_tokenize, nltk.word_tokenize

**Tokenization using NLTK: ** from nltk.tokenize import wordpunct_tokenize (Splits for Punctuations too), from nltk.tokenize import TreebankWordTokenizer (Full Stop won't be considered as a word in in-between places)

**2. Stemming: ** from nltk.stem import PorterStemmer

**Issue:** (With PorterStummer) Some words loose it's form E.g. history---->histori; congratulations --> congratuli

B) from nltk.stem import RegexpStemmer;   reg_stemmer=RegexpStemmer('ing$|s$|e$|able$', min=4) [eating --> eat)

C) from nltk.stem import SnowballStemmer; Performs better than PorterStemmer;  from nltk.stem import SnowballStemmer; 

Example: stemming.stem("fairly"),stemming.stem("sportingly") --> ('fairli', 'sportingli')

snowballsstemmer.stem("fairly"),snowballsstemmer.stem("sportingly") --> ('fair', 'sport')

**Snowball Stemmer better than Porter Stemmer**

**3. Lemmatization: ** Stemming - Word Stems; Lemmatization --> Lemma (Root Words rather than Word Stems) (Done using WordNetCorpusReader); lemmatizer.lemmatize("going",pos='v') [Better for Verbs]

**NLTK provides WordNetLemmatizer class which is a thin wrapper around the wordnet corpus. This takes more time because it gets word from the Corpus**

**4. StopWords**: Few Words like "I, the, of" don't play much in Model Buidling for Sentimental Analysis/Spam Classification; We will use Tokenization to take sentence tokens(Para to Sentence Tokenization), We will see stopwords, filter and then use Stemming/lemmatization only for those important words

from nltk.corpus import stopwords;

stopwords.words('english') --> Shows all Stopwords in English Language

**5. Parts of Speech Tagging:** lemmatizer.lemmatize("going",pos='v') [pos - Parts of Speech Tagging] We took a sentence, removed stopwords and saw the POS_TAG for each tokens

print(nltk.pos_tag("Taj Mahal is a beautiful Monument".split()))

**6. Named Entity Recognition:** Provides like a Image from POS_TAG

words=nltk.word_tokenize(sentence); tag_elements=nltk.pos_tag(words); nltk.ne_chunk(tag_elements).draw()

**7. One-Hot Encoding: ** Converts each and every vocabulary into vectors

**8. Bag of Words:** Lower all the words (To avoid Repetition) --> Apply Stopwords (He, She will be deleted); Select Vocabulary and Frequency (OHE was for each word; BOW gets vectors for each sentences level. 

Two Types: (i) Binary BOW --> If frequency is 2, it reduces to 1 ([2 1 0] --> [1 1 0]) It can have only 0 or 1
          (ii) BOW --> Can have frequencies

**Imported Using "from sklearn.feature_extraction.text import CountVectorizer"**

cv=CountVectorizer(max_features=100) --> Takes top 100 features with Maximum Frequency


