# Natural-Language-Processing-Skills

#### Using Multinomial Naive Bayes in NLP Problem, as it works well with Text Data

**Natural Language Processing using Machine Leaning:**

**NLP Libraries: NLTK, spaCy, Gensim**

**My Skills in NLP:**

**1. Tokenization**

**2. Stemming**

**3. Lemmatization**

**4. Stop Words**

**5. Parts of Speech Tagging**

**6. Named Entity Recognition**

**7. One-Hot Encoding**

**8. Bag of Words**

**9. N Grams**

**10.TF-IDF**

**11.CBOW - Word2Vec**

**12. Skipgram - Word2Vec**

**13. Avg Word2Vec**

**14. Spam Ham Classification Project using BOW and TF-IDF**

**NLP Projects Workflow:**

(i) Dataset Loading

(ii) Text Pre-Processing - I --> Tokenization, Lowercase the words, Regular Expressions

## Create the Bag OF Words model
from sklearn.feature_extraction.text import CountVectorizer
## for Binary BOW enable binary=True
cv=CountVectorizer(max_features=2500,ngram_range=(1,2))

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

For Binary BOW --> cv=CountVectorizer(max_features=100,binary=True)

**9. N Grams:** [Combined with Bag of Words] Bigrams (2 Words Combination); Trigrams (3 Words Combination)

**sklearn has n grams feature** --> (1,1) - Unigram; (1,2) - Unigram, Bigram; (1,3) - Unigram, Bigram, Trigram; (2,3) - Bigram, Trigram

cv=CountVectorizer(max_features=100,binary=True,ngram_range=(2,3))

**Start with (1,1); Try with (1,2), (1,3); Then try with Hyperparameter like Max_features; Then finally go for Bigram and Trigram**

**10. TF-IDF (Term Frequency - Inverse Document Frequency)**

Term Frequency (TF) = Number of Repetition of Words in Sentence / Number of words in sentence

Inverse Document Frequency (IDF) = log (No. of sentences/No. of sentences containing the word)

We calculate TF, IDF separately and multiply both, for each sentences and finally it comes as a Vector.

** Imported from - sklearn (from sklearn.feature_extraction.text import TfidfVectorizer)**

Also works with n-Grams

**Gives Word Importance - BOW calculates only 0's and 1's; TF-IDF, we calculate TF and multiply with IDF, so we get a vector like, where word importance is calculated**


**Notes: Word Embeddings - Convert Words to vectors, that maintains meaning and similar words are placed together (Done through PCA Example)**

**Word Embeddings Types:**

**(i) Count or Frequency - One-Hot Encoding, Bag of Words, TF-IDF**

**(ii) Deep Learning Trained Model - Word2Vec (Two Types: Continuous Bag of Words (CBOW) and Skipgram**

**Word to Vec:** Words are converted to Feature Representation; 

**Each Vocabulary (Unique Word) will be represented as n-dimension feature representation as a Vecor**

**Google Trained a Deep Learning Model on 3Bn Words**

**Recommendations happen on based of it [Distance = 1- Cosine Similarity (1- cos(theta) between the Vectors)**

**11. CBOW - Word2Vec** - We have Window Size; Take Window Size as Odd; In general, we take the Middle word and train it with both before and after words, so that is is aware of words. **It is kind of ANN Deep Learning Architecture**

We then do One-Hot Encoding for all words; Take the Inputs as Input Layer

Window Size in the Hidden Layer

We know the Output Word and Train according to it. Each and every dimension in a vector is connected with every dimension in Hidden and Output Layer

**If Window size is 5, I will get Vector of 5 Dimnensions in Output Layer**

**Window Size --> Feature Representation**

**Google Model said to have 300 Features means, 300 Window Size. More the Window Size, better the Model**

**12. Skipgram - Word2Vec** - In Skipgram also it will be similar to CBOW, but the thing is we change the Input and Output Vice-Versa.

**Input Layer Size - No. of Words in Vocabulary**

**Hidden Layer Size - Window Size (Feature Representation)**

**Output Layer uses a Softmax (As in ANN Architecture)**

**Output Layer - (Number of Neurons will be Number of words in the dataset (E.g. iNeuron Company Related to - Means 4 Neurons); Output will be 7 Vectors**

**13. Avg Word2Vec:** In Word2Vec, we found of Vectors for each and every word. Say for example of Google's Pretrained model which has 300 Dimensions, It has vector of 300 Dimensions for each and every word. (Which won't be much helpful)

So, In Avg, Word2Vec, we take all dimensions of Vectors and take it's avergae.

**Finally, we get only one final vector at the sentence level of 300 Dimensions, which we can use and train the model**

**Implementation using Gensim:**

**Using Google's Pretrained Model:**

**Similarity Score is calculated using Cosine Similarity**

Model Name - 'word2vec-google-news-300'

wv = api.load('word2vec-google-news-300')

vec_king.shape --> 300 Dimensions

wv.most_similar('cricket') --> We get Cricketing, Cricketers

wv.most_similar('happy') --> We get Glad, Pleased

wv.similarity("hockey","sports") --> We get Similarity Score

vec=wv['king']-wv['man']+wv['woman']; After this once we check similar words for 'vec', we get Queen in it (Next to King, as King is present in the sentence itself)

**14. Spam Ham Classification Project using BOW and TF-IDF**

#### Using Multinomial Naive Bayes in NLP Problem, as it works well with Text Data

## Create the Bag OF Words model

from sklearn.feature_extraction.text import CountVectorizer

## for Binary BOW enable binary=True

cv=CountVectorizer(max_features=2500,ngram_range=(1,2))

from sklearn.naive_bayes import MultinomialNB

spam_detect_model=MultinomialNB().fit(X_train,y_train)

y_pred=spam_detect_model.predict(X_test)

**TF-IDF Model:**

from sklearn.feature_extraction.text import TfidfVectorizer

tv=TfidfVectorizer(max_features=2500,ngram_range=(1,2))

X_train=tv.fit_transform(X_train).toarray()

X_test=tv.transform(X_test).toarray()

tv.vocabulary_

from sklearn.naive_bayes import MultinomialNB

spam_tfidf_model = MultinomialNB().fit(X_train, y_train)

**Prediction:** y_pred=spam_tfidf_model.predict(X_test)

score=accuracy_score(y_test,y_pred); print(score)


