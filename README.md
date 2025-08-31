# Natural-Language-Processing-Skills

#### Text Data Pre-Processing: 

## Libraries used: import re; from nltk.corpus import stopwords; import nltk; from bs4 import BeautifulSoup

### For HTML used BeautifulSoup

## Krish had some issues and later gave some extra spaces; Later it worked for all the regular expressions he gave 

## (i) Converting Lower Case - df['reviewText']=df['reviewText'].str.lower()

## (ii) Pre-Processing using Lambda Codes - 

(a) Removing special characters - df['reviewText']=df['reviewText'].apply(lambda x:re.sub('[^a-z A-z 0-9-]+', '',x))

(b) Remove the stopswords - df['reviewText']=df['reviewText'].apply(lambda x:" ".join([y for y in x.split() if y not in stopwords.words('english')]))

(c) Remove url df['reviewText']=df['reviewText'].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(x)))

(d) Remove html tags - df['reviewText']=df['reviewText'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())

(e) Remove any additional spaces - df['reviewText']=df['reviewText'].apply(lambda x: " ".join(x.split()))

## (iii) Lemmatization - from nltk.stem import WordNetLemmatizer; lemmatizer=WordNetLemmatizer()

def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()]); df['reviewText']=df['reviewText'].apply(lambda x:lemmatize_words(x))

## (iv) Train-Test Split - X_train, X_test, y_train, y_test = train_test_split(df['review_text'], df['rating'], test_size=0.2, random_state=42)

### Very Important: .toarray() is used to convert it to Arrays

## (v) Convert words or sentences into vectors using Bag of Words (BoW) and TF-IDF - from sklearn.feature_extraction.text import CountVectorizer; bow = CountVectorizer(); X_train_bow = bow.fit_transform(X_train).toarray(); X_test_bow = bow.transform(X_test).toarray()

## Same for TF-IDF - from sklearn.feature_extraction.text import TfidfVectorizer; tfidf = TfidfVectorizer(); X_train_tfidf = tfidf.fit_transform(X_train).toarray(); X_test_tfidf = tfidf.transform(X_test).toarray()

## (vi) Model Training - We will use a Naive Bayes classifier, which usually performs well with sparse matrices. We can choose GaussianNB or MultinomialNB. Here, we will use GaussianNB: from sklearn.naive_bayes import GaussianNB [We can also use Multinomial Naive Bayes too]

## For BOW: nb_model_bow = GaussianNB(); nb_model_bow.fit(X_train_bow, y_train)

## Similarly, we create a Naive Bayes model for TF-IDF: nb_model_tfidf = GaussianNB(); nb_model_tfidf.fit(X_train_tfidf, y_train)

## (vii) Prediction on Test Data: y_pred_bow = nb_model_bow.predict(X_test_bow); y_pred_tfidf = nb_model_tfidf.predict(X_test_tfidf)

## (viii) Model Evaluation: Evaluating the models using accuracy score, confusion matrix, and classification report: from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

## Bag of Words evaluation: print("Bag of Words Accuracy:", accuracy_score(y_test, y_pred_bow)); print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bow)); print("Classification Report:\n", classification_report(y_test, y_pred_bow))

## TF-IDF evaluation: print("TF-IDF Accuracy:", accuracy_score(y_test, y_pred_tfidf)); print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tfidf)); print("Classification Report:\n", classification_report(y_test, y_pred_tfidf))

## For this dataset, Bag of Words gives an accuracy of ~58%, and TF-IDF gives an accuracy of ~58.1%, which is not very high. This is because the dataset is quite large and sparse.

## The confusion matrix shows how many predictions were correct or incorrect. For large datasets, Word2Vec typically performs better because it captures semantic meaning, whereas Bag of Words and TF-IDF only capture word frequency.

### Best Practices for NLP Projects:

(i) Data Preprocessing And Cleaning

(ii) Train Test Split

(iii) BOW,TFIDF,Word2vec

(iv) Train ML algorithms

(v) Prediction and Evaluation

### Data Leakage is like I am going to write a exam and while going to exam itself I know the questions which is going to come

#### Using Multinomial Naive Bayes in NLP Problem, as it works well with Text Data

#### In our Kindle Review Sentiment Analysis, we had around 5 categories and we converted into 0'2 and 1's using Lambda function and later it had fairly equal categories, of not being of much imbalanced dataset

#### Krish Made a Mistake in Spam-Ham Classification Project: Usually, preprocessing and cleaning of text data are performed on the entire dataset. Then, Bag of Words or TF-IDF converts all sentences into vectors on the full dataset before we do a train-test split. This sequence introduces a major problem because the model indirectly gains information about the test set through the vocabulary created on the full dataset. To prevent this data leakage, we must first perform the train-test split and then apply Bag of Words or TF-IDF to convert sentences into vectors separately for the training and test sets. This ensures that the test set remains completely unseen during training, akin to preparing for an exam without ever seeing the questions in advance.



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

**15. Best Practises For Solving ML Problems**

**16. Part 1 - Spam Ham Classification With Word2vec And AvgWord2vec**

**17. Part 2 - Spam Ham Classification With Word2vec And AvgWord2vec**

**18. Part 1-Kindle Review Sentiment Analysis**

**19. Part 2- Kindle Review Sentiment Analysis**

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

**15. Best Practises For Solving ML Problems**

In this lecture, we are going to continue the discussion on natural language processing, particularly focusing on spam classification. In the previous video, we demonstrated how to perform spam classification using Bag of Words and TF-IDF techniques. The entire workflow started with applying Bag of Words using CountVectorizer, where we obtained the vocabulary and converted all sentences into vectors. After this, we performed a train-test split to separate the data into training and testing sets. We then trained our model and evaluated it using accuracy and other classification metrics. Similarly, for the TF-IDF model, we transformed the sentences into vectors, split the dataset, trained the model, and evaluated its performance.

Now, there is an extremely important best practice that we need to discuss, which was deliberately not mentioned in the previous video. The mistake we made previously is related to the order of operations between vectorization (Bag of Words or TF-IDF) and train-test splitting. Usually, preprocessing and cleaning of text data are performed on the entire dataset. Then, Bag of Words or TF-IDF converts all sentences into vectors on the full dataset before we do a train-test split. This sequence introduces a major problem because the model indirectly gains information about the test set through the vocabulary created on the full dataset. To prevent this data leakage, we must first perform the train-test split and then apply Bag of Words or TF-IDF to convert sentences into vectors separately for the training and test sets. This ensures that the test set remains completely unseen during training, akin to preparing for an exam without ever seeing the questions in advance.

Let us now implement this corrected approach. First, we start by performing the train-test split on the input corpus and target variable.

"from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=42)"

Here, corpus represents our cleaned text data, and y is the target variable indicating spam or non-spam. We have now correctly separated the training and test sets. The next step is to apply Bag of Words on the training data. We initialize the CountVectorizer with the desired max_features.

"from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000)
X_train = cv.fit_transform(X_train).toarray()"

Notice that we use fit_transform only on the training data to create the vocabulary. For the test data, we do not fit again; instead, we only transform it using the existing vocabulary created from the training set.

"X_test = cv.transform(X_test).toarray()"

By doing this, we ensure that our test data does not influence the vocabulary, thereby preventing data leakage. At this stage, our training and test data are properly vectorized and ready for model training. We can now apply a machine learning classifier, for instance, Multinomial Naive Bayes.

"from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))"

With this approach, you typically achieve an accuracy around 98%, which is both high and reliable due to the correct separation of training and test sets.

Similarly, for the TF-IDF model, we follow the same corrected process. First, initialize the TfidfVectorizer and fit it on the training data.

"from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(X_train).toarray()
X_test = tfidf.transform(X_test).toarray()"

Again, we use fit_transform on the training data and only transform on the test data. This ensures that the test set remains unseen during training. After vectorization, we train the classifier, such as Multinomial Naive Bayes, and evaluate the results.

"model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('TF-IDF Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))"

This method typically yields an accuracy around 97%–98%, confirming that the model performs well without data leakage. The key takeaway from this lecture is the importance of performing the train-test split before applying vectorization techniques such as Bag of Words or TF-IDF. This practice is crucial not just for spam classification but for any machine learning problem involving feature extraction from the entire dataset. Always ensure that preprocessing, train-test splitting, and vectorization are done in the correct order to avoid leaking information from the test set into the training process.

By following these best practices, you maintain the integrity of the machine learning workflow and ensure that your models generalize well to unseen data. This concludes the lecture, and in the next video, we will continue exploring advanced NLP techniques while adhering to these best practices.


### **16. Part 1 - Spam Ham Classification With Word2vec And AvgWord2vec**

Hello guys. So we are going to continue our discussion with respect to natural language processing. In this video, we are going to solve the spam Ham classification project using word2vec and average word2vec. I'll just change the spelling. Okay, so this will be an important one guys, because with this, the accuracy level will be quite good, you know? But the main thing is that how we can train this entire project from scratch using word2vec; that is what we are going to see. And for this, we are going to use Gensim as the library. So again, you know how to install this. You just need to write "pip install gensim" if you don't have it already.

Next, I’m going to import two things from gensim: "from gensim.models import Word2Vec, KeyedVectors". I have already shown you how you can load the Google News 300 model, which is a pre-trained Word2Vec model. By default, this particular Word2Vec model will give you vectors. If I probably check the shape, it will give 300-dimensional vectors. For example, if I want to see the vector of the word "king", the "king" vector will be these 300-dimensional values.

Now our plan is to take the entire dataset, which consists of all the sentences, and create a Word2Vec model from scratch. This will be amazing because it will allow you to perform many tasks with it. In this, we are also going to use different techniques. First of all, I’m making some changes in NLTK: I will not apply the Porter stemmer, but I’ll apply the Lemmatizer. So I just import "from nltk.stem import WordNetLemmatizer" and initialize it with "lemmatizer = WordNetLemmatizer()".

The next step is preprocessing the messages. I’m removing all special characters using regular expressions, converting the text to lowercase, and splitting the sentences. I’m not removing stop words because I want to see how each word will get its vector. So every word in the review is lemmatized and appended to the corpus. If there’s an error, it’s probably because I didn’t import re, so I import the regular expression module. I also import NLTK and download all stopwords with "nltk.download('stopwords')" if needed. After executing this, I have my entire corpus prepared, with all words lemmatized.

There are multiple ways to process the text. For example, you can use the function simple_preprocess from Gensim. I import it using "from gensim.utils import simple_preprocess" and also import "from nltk.tokenize import sent_tokenize". The simple_preprocess function converts a document into a list of lowercase tokens, ignoring tokens that are too short or too long. This function will automatically handle lowercasing and tokenization. I create a list of words by iterating over sentences in the corpus and applying "words = [simple_preprocess(sentence) for sentence in corpus]". This gives all words for each sentence, and I can check each list of words for verification.

Once I have all the words, the next step is to train the Word2Vec model from scratch. I import Word2Vec from Gensim using "from gensim.models import Word2Vec". Now, to train the model: "model = Word2Vec(sentences=words, vector_size=100, window=5, min_count=1)". Here, sentences=words is the input corpus, vector_size=100 is the number of dimensions for each word vector, window=5 defines the maximum distance between a target word and neighboring words, and min_count=1 ignores all words with frequency lower than 1 to build the vocabulary. You can adjust vector size to 100, 200, or 300 depending on your needs. This way, every word in the corpus is converted into a 100-dimensional vector, and you can proceed to calculate average word2vec representations for your sentences or documents.

Okay, so let me just go ahead and probably execute this. I will store the entire trained Word2Vec model in a variable called "model". This variable will now hold my trained model. If I write "model.wv", that is Word2Vec, and then "model.wv.index_to_key", this line of code retrieves all the vocabulary from the trained model. If I execute it, I can see all the words that are part of the vocabulary. To check the vocabulary size, I can write "model.corpus_count". Executing this, you can see that my vocabulary size is 5569. Additionally, the model shows how many epochs it has been trained for; by default, it trains for five epochs. Increasing the number of epochs improves the quality of training, but longer training times may be required if we increase epochs to 100 or more. For demonstration purposes, I have trained for the default five epochs.

Next, to find words similar to a given word, I can use "model.wv.most_similar('skid')". This returns words similar to "skid" along with their cosine similarity scores. Similarly, for the word "good", I can check similar words such as "morning", "what", "not", "happy", etc. Since the training was done for only five epochs, the similarity may not be perfect, but you can already observe some reasonable results. To examine the vector for a word, I can write "model.wv['good']" and check its size. You’ll notice it is 100, confirming that each word is represented by a 100-dimensional vector. Using "model.wv['good'].shape" will also show the 100 dimensions for each word vector.

Now, the super important concept is why we use average Word2Vec. Suppose my first sentence is "Go until Jurong Point" and each word in this sentence has a 100-dimensional vector from Word2Vec. For example, "go" has a 100-dimensional vector, "until" has a 100-dimensional vector, and "Jurong" has a 100-dimensional vector. But for the entire sentence, I need a single 100-dimensional vector. What we do is take all the word vectors in the sentence and average them. This gives one vector of 100 dimensions representing the entire sentence. This is the intuition behind average Word2Vec. The sentence vector becomes the input for our model, and the output is the label associated with the sentence, such as 0 or 1.

To implement this in Python, I created a function called "average_word2vec": "def average_word2vec(doc, model): return np.mean([model.wv[word] for word in doc if word in model.wv.index_to_key], axis=0)". This function calculates the row-wise mean of all word vectors in a sentence. Before applying this, I also want to track the progress using "tqdm". So I import it: "from tqdm import tqdm". Now, I apply the "average_word2vec" function on every sentence in my dataset. This computes the average of all word vectors for each sentence and provides a single vector representing the sentence.

While executing this, I realized I hadn’t imported NumPy, so I import it: "import numpy as np". After applying the function, I check my variable x, which now contains the averaged vectors for each sentence. For example, x[0] gives the first sentence vector, x[1] the second, and so on. Each vector has 100 dimensions. To convert these into a NumPy array, I write "x_new = np.array(x)". Now x_new becomes my independent features array. The number of sentences is 5569, corresponding to the input size of the dataset. Each vector in x_new is 100-dimensional.

Finally, I define the dependent feature, which is the output label for each sentence. I assign this to "y", taking the label column from the dataset. Now I have my independent features in x_new and dependent features in y. In the next video, we will use these features to train a machine learning algorithm. The input features are ready, and the output labels are defined, so all that remains is to train the model. This process demonstrates how average Word2Vec vectors are generated for each sentence, giving a compact, fixed-size representation suitable for training a classifier.

**17. Part 2 - Spam Ham Classification With Word2vec And AvgWord2vec**

So we are going to continue the discussion with respect to the Spam Ham project using Word2Vec and average Word2Vec. At this point, we have already converted all the sentences into vectors, and we have x_new, which is our input data. It contains 5569 rows. Before feeding this dataset to the training algorithm, there are a few important things to understand. Each sentence vector has 100 dimensions, which is perfect. However, when I check the original messages dataset, it has 5572 rows and two columns. There is a mismatch because after preprocessing and converting sentences into vectors using average Word2Vec, x_new has only 5569 rows. Therefore, the independent feature set and the output labels are not aligned.

The discrepancy occurs because three rows were lost during preprocessing. To investigate this, I checked the corpus before vectorization. I added a line to get the length of each sentence in the corpus using Python’s zip function: "for i, j, k in zip(list(map(len, corpus)), corpus, messages)". Then I displayed only those messages where the length of the sentence was less than one. Executing this, I found three messages that had become blank. This happened because the preprocessing step kept only alphabetical characters (A–Z, a–z) and removed everything else, including numbers and special characters. As a result, these sentences were reduced to empty strings, causing the row loss.

To fix the issue with the dependent variable y, I filtered the messages similarly: "y = [msg for msg, sentence in zip(messages, corpus) if len(sentence) > 0]". This ensures that y contains only labels corresponding to non-empty sentences. Now, both x_new and y have matching lengths of 5569. After executing this, x_new contains all the sentence vectors, each of 100 dimensions, and y contains the output labels aligned correctly. We also verified the type and length of x; it is a list of arrays with length 5569. Accessing x[0], x[1], or x[2] gives the first, second, and third sentence vectors, respectively.

The next crucial step is to convert these vectors into a structured format, specifically a Pandas DataFrame, to create independent features for machine learning models. Each vector should occupy one row in the DataFrame, corresponding to one sentence. For example, x[0] represents the first sentence vector and should be reshaped into a row. To achieve this, I use the following code: "df = pd.DataFrame(); for i in range(len(x)): df = df.append(pd.DataFrame(x[i].reshape(1, -1)))". Here, x[i].reshape(1, -1) reshapes the 100-dimensional vector into a single row, and then we append it to the DataFrame. x[0] is initially a one-dimensional array of shape (100,), and reshaping it allows it to be inserted as a row in the DataFrame. Repeating this for all vectors ensures that each sentence vector occupies a separate row, creating the final independent feature set for training.

Once I write x[i].reshape(1, -1) in the code, the vector for a single sentence will be reshaped into a row with one row and 100 columns. This is exactly what we need because each sentence, after averaging Word2Vec vectors, becomes a data point represented by 100 features. The first sentence corresponds to the first row, the second sentence to the second row, and so on. To add all sentence vectors from x into a DataFrame, I use a for loop: "for i in range(len(x)): df = df.append(pd.DataFrame(x[i].reshape(1, -1)), ignore_index=True)". The ignore_index=True ensures that the DataFrame reindexes rows automatically, preventing duplicate index issues. Executing this loop may take some time, depending on the size of the dataset. After the loop finishes, calling "df.shape" shows that the DataFrame contains 5569 rows and 100 columns, corresponding to all sentences and their 100-dimensional vectors. Inspecting "df.head()" displays the first few sentence vectors. At this point, the DataFrame df serves as our independent feature set, so we can assign it to x using "x = df". The dependent variable y has already been prepared previously, so we can print it to confirm its values.

The next step is performing a train-test split. Using sklearn’s train_test_split, I split x and y into training and testing sets. After splitting, X_train and y_train contain the training data, while X_test and y_test contain the testing data. At this stage, we have our independent and dependent features ready for machine learning. For this project, since the vectors are dense, I chose to use a Random Forest classifier from sklearn. I import it using "from sklearn.ensemble import RandomForestClassifier" and initialize it: "classifier = RandomForestClassifier()". Then, I attempt to fit the classifier on the training data using "classifier.fit(X_train, y_train)".

At this point, I encountered an error: "Input contains NaN, infinity, or a value too large for dtype". This indicates that some feature vectors contain missing values. To check, I tried "x.isnull().sum()" and found 12 null values spread across different features. To resolve this, I added the dependent variable y as a new column to the DataFrame using "df['Output'] = y". Then, I removed all rows containing null values with "df.dropna(axis=0, inplace=True)". After this, "df.isnull().sum()" confirmed that there were no missing values remaining. Now, x corresponds to all independent features (df.drop('Output', axis=1)) and y is the output feature (df['Output']).

With clean data, I performed the train-test split again and fitted the Random Forest classifier successfully. Once the model is trained, predictions can be made on the test set using "y_pred = classifier.predict(X_test)". To evaluate performance, I computed the accuracy score with "accuracy_score(y_test, y_pred)" and printed the classification report using "classification_report(y_test, y_pred)". Both metrics showed good performance. However, because Random Forest can overfit, hyperparameter tuning may be necessary to improve generalization.

This process demonstrates how to handle all practical issues that arise when preparing a dataset for machine learning with Average Word2Vec. Line by line, we reshaped vectors, constructed a DataFrame, removed missing values, performed a train-test split, trained a Random Forest classifier, and evaluated its performance. Practicing this workflow on different datasets will reinforce understanding and enable solving a variety of NLP classification problems using word embeddings.

**18. Part 1-Kindle Review Sentiment Analysis**

So we are going to continue the discussion with respect to natural language processing. In this video, I'm going to start a new project called Kindle Review Sentiment Analysis. An amazing use case altogether. There are many things we can do here, but first, let's talk about the dataset.

The dataset is a small subset of book reviews from Amazon Kindle stores, specifically the "Books" category. The core dataset contains product reviews from Amazon Kindle store categories spanning May 1996 to 2014. Each reviewer has at least five reviews, and each product has at least five reviews in the dataset. One very important feature is the output feature called rating. The dataset is taken from Amazon product data via Julian McAuley at UCSD. The license belongs to them, so the entire credit is theirs.

Now, what can we do with this dataset? We can perform sentiment analysis, detect fake reviews or outliers, or even explore advanced techniques like Word2Vec embeddings. To get started, let's import pandas and read the dataset. import pandas as pd; df = pd.read_csv("Kindle_reviews/Kindle_Reviews.csv"); df.head()

This gives us a view of how the dataset looks. The most important features for this use case are review_text and rating. We can isolate these features as follows: df = df[['review_text', 'rating']]; df.head()

Next, let's check the dataset shape and any missing values: df.shape; df.isnull().sum()

Luckily, there are no missing values. We can also explore how many unique ratings exist: df['rating'].unique(); df['rating'].value_counts()

There are five possible ratings, and the counts indicate that the dataset is fairly balanced. This makes it suitable for training models.

### The first step in building a sentiment analysis model is preprocessing and cleaning, which can also be considered feature engineering. The ratings need to be converted to sentiment labels. For simplicity, we will consider reviews with ratings less than 3 as negative and greater than or equal to 3 as positive.

df['rating'] = df['rating'].apply(lambda x: 0 if x < 3 else 1); df.head(); df['rating'].unique()

Now our dataset has only two classes: 0 for negative reviews and 1 for positive reviews. Checking the counts: df['rating'].value_counts()

### We have approximately 8,000 positive and 4,000 negative reviews, which is a good balance for most machine learning algorithms.

#### We had so many categories and we reduced to 0's and 1's

So far, we have successfully loaded the dataset, selected relevant features, handled missing values, and transformed the ratings into binary sentiment labels. The next steps will involve train-test split, feature extraction using Bag of Words, TF-IDF, or Word2Vec, and training machine learning algorithms on this dataset. Remember, the best practice is to perform preprocessing before splitting the dataset to avoid data leakage.

Now, this is the first thing that we specifically did. Next, I will go ahead and write some preprocessing steps that we usually apply to any text data. The first step is to convert all text to lowercase, which is important for consistency. We can do this using the str.lower() function in pandas:

df['review_text'] = df['review_text'].str.lower()
df.head()

### This converts all words in the review_text column to lowercase, making text processing easier.

The second step is to clean the text by removing unwanted characters. This includes special characters, punctuations, URLs, email addresses, and HTML tags. We can use regular expressions, stopwords removal, and the BeautifulSoup library for this purpose. First, we import the necessary libraries:

import re
from nltk.corpus import stopwords
import nltk
from bs4 import BeautifulSoup

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

To remove special characters, we can use a lambda function with regex:

df['review_text'] = df['review_text'].apply(lambda x: re.sub(r'[^A-Za-z0-9 ]+', ' ', x))


Next, we remove stopwords:

df['review_text'] = df['review_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))


To remove URLs and emails, we can apply the following regex:

df['review_text'] = df['review_text'].apply(lambda x: re.sub(r'\S*@\S*\s?', '', x))  # Remove emails
df['review_text'] = df['review_text'].apply(lambda x: re.sub(r'http\S+|https\S+|www\S+|ftp\S+|ssh\S+', '', x))  # Remove URLs


To remove HTML tags, we can use BeautifulSoup:

df['review_text'] = df['review_text'].apply(lambda x: BeautifulSoup(x, "lxml").get_text())


Finally, we remove extra whitespaces to clean up the text:

df['review_text'] = df['review_text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
df.head()


Now that our text is cleaned, the next step is lemmatization, which reduces words to their base form. We can use the WordNetLemmatizer from NLTK:

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

df['review_text'] = df['review_text'].apply(lambda x: lemmatize_words(x))
df.head()


This will take some time as lemmatization is computationally expensive, but it helps standardize the text for modeling.

Once preprocessing is complete, the next step is to perform the train-test split. We use train_test_split from sklearn.model_selection for this:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['review_text'], df['rating'], test_size=0.2, random_state=42)


Now, we have preprocessed text data ready for modeling and have split it into training and testing sets. This completes the key preprocessing and train-test split steps while following best practices for NLP tasks.          

### **19. Part 2- Kindle Review Sentiment Analysis**

We are going to continue our discussion with respect to the Kindle Review Sentiment Analysis project, and we are now in part two. In part one, we already completed preprocessing and cleaning and also performed a train-test split.

According to the methodologies we are following, the next step is to convert words or sentences into vectors using Bag of Words (BoW) and TF-IDF. Later, we will apply ML algorithms, and as an assignment, you can try Word2Vec. Make sure to use average Word2Vec to convert sentences into vectors effectively.

First, let’s import the required libraries and create a Bag of Words vectorizer:

from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer()
X_train_bow = bow.fit_transform(X_train).toarray()
X_test_bow = bow.transform(X_test).toarray()


Here, we apply fit_transform on the training data and transform on the test data to avoid data leakage. Similarly, we can create a TF-IDF vectorizer:

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()


Now, both X_train_bow and X_train_tfidf contain the independent features for our ML model. You can verify them by inspecting a few rows:

X_train_bow[:5]
X_train_tfidf[:5]


Next, we will use a Naive Bayes classifier, which usually performs well with sparse matrices. We can choose GaussianNB or MultinomialNB. Here, we will use GaussianNB [We can also use Multinomial Naive Bayes too]:

from sklearn.naive_bayes import GaussianNB

nb_model_bow = GaussianNB()
nb_model_bow.fit(X_train_bow, y_train)


Similarly, we create a Naive Bayes model for TF-IDF:

nb_model_tfidf = GaussianNB()
nb_model_tfidf.fit(X_train_tfidf, y_train)


Once the models are trained, we can make predictions on the test set:

y_pred_bow = nb_model_bow.predict(X_test_bow)
y_pred_tfidf = nb_model_tfidf.predict(X_test_tfidf)


Now, let’s evaluate the models using accuracy score, confusion matrix, and classification report:

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Bag of Words evaluation
print("Bag of Words Accuracy:", accuracy_score(y_test, y_pred_bow))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bow))
print("Classification Report:\n", classification_report(y_test, y_pred_bow))

# TF-IDF evaluation
print("TF-IDF Accuracy:", accuracy_score(y_test, y_pred_tfidf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tfidf))
print("Classification Report:\n", classification_report(y_test, y_pred_tfidf))


For this dataset, Bag of Words gives an accuracy of ~58%, and TF-IDF gives an accuracy of ~58.1%, which is not very high. This is because the dataset is quite large and sparse.

## The confusion matrix shows how many predictions were correct or incorrect. For large datasets, Word2Vec typically performs better because it captures semantic meaning, whereas Bag of Words and TF-IDF only capture word frequency.

As an assignment, you should try implementing Word2Vec with average embeddings for all sentences and train the same Naive Bayes or other ML models. This will likely improve accuracy significantly.
