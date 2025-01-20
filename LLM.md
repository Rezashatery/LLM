## What is an LLM?
An LLM is a neural network designed to understand, generate, and respond to human-
like text. These models are deep neural networks trained on massive amounts of text
data, sometimes encompassing large portions of the entire publicly available text on
the internet.
The “large” in “large language model” refers to both the model’s size in terms of
parameters and the immense dataset on which it’s trained. Models like this often have
tens or even hundreds of billions of parameters, which are the adjustable weights in
the network that are optimized during training to predict the next word in a sequence.
Next-word prediction is sensible because it harnesses the inherent sequential nature
of language to train models on understanding context, structure, and relationships
within text. LLMs utilize an architecture called the transformer, which allows them to pay selective attention to different parts of the input when making predictions, making them
especially adept at handling the nuances and complexities of human language.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image1.png?raw=true)

Since LLMs are capable of generating text, LLMs are also often referred to as a form
of generative artificial intelligence, often abbreviated as generative AI or GenAI. As illus-
trated in figure 1.1, AI encompasses the broader field of creating machines that can
perform tasks requiring human-like intelligence, including understanding lan-
guage, recognizing patterns, and making decisions, and includes subfields like machine
learning and deep learning.

The spam classification example, in traditional machine learning,
human experts might manually extract features from email text such as the fre-
quency of certain trigger words (for example, “prize,” “win,” “free”), the number of
exclamation marks, use of all uppercase words, or the presence of suspicious links.
This dataset, created based on these expert-defined features, would then be used to
train the model. In contrast to traditional machine learning, deep learning does not
require manual feature extraction. This means that human experts do not need to
identify and select the most relevant features for a deep learning model. (However,
both traditional machine learning and deep learning for spam classification still
require the collection of labels, such as spam or non-spam, which need to be gath-
ered either by an expert or users.)


## Applications of LLMs

Owing to their advanced capabilities to parse and understand unstructured text data,
LLMs have a broad range of applications across various domains. Today, LLMs are
employed for machine translation, generation of novel texts (see figure 1.2), senti-
ment analysis, text summarization, and many other tasks. LLMs have recently been
used for content creation, such as writing fiction, articles, and even computer code.
Moreover, LLMs may be used for effective knowledge retrieval from vast volumes
of text in specialized areas such as medicine or law. This includes sifting through doc-
uments, summarizing lengthy passages, and answering technical questions.
In short, LLMs are invaluable for automating almost any task that involves parsing
and generating text. Their applications are virtually endless, and as we continue to
innovate and explore new ways to use these models, it’s clear that LLMs have the
potential to redefine our relationship with technology, making it more conversational,
intuitive, and accessible.

## Stages of building and using LLMs
Why should we build our own LLMs? Coding an LLM from the ground up is an excel-
lent exercise to understand its mechanics and limitations. Also, it equips us with the
required knowledge for pretraining or fine-tuning existing open source LLM architec-
tures to our own domain-specific datasets or tasks.Using custom-built LLMs offers several advantages, particularly regarding data pri-
vacy. For instance, companies may prefer not to share sensitive data with third-party
LLM providers like OpenAI due to confidentiality concerns. Additionally, developing
smaller custom LLMs enables deployment directly on customer devices, such as laptops
and smartphones, which is something companies like Apple are currently exploring. 
This local implementation can significantly decrease latency and reduce server-related
costs. Furthermore, custom LLMs grant developers complete autonomy, allowing
them to control updates and modifications to the model as needed.
This local implementation can significantly decrease latency and reduce server-related
costs. Furthermore, custom LLMs grant developers complete autonomy, allowing
them to control updates and modifications to the model as needed.
The general process of creating an LLM includes pretraining and fine-tuning. The
“pre” in “pretraining” refers to the initial phase where a model like an LLM is trained
on a large, diverse dataset to develop a broad understanding of language. This pre-
trained model then serves as a foundational resource that can be further refined
through fine-tuning, a process where the model is specifically trained on a narrower
dataset that is more specific to particular tasks or domains. This two-stage training
approach consisting of pretraining and fine-tuning is depicted in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image2.png?raw=true)

**NOTE** Readers with a background in machine learning may note that label-
ing information is typically required for traditional machine learning models
and deep neural networks trained via the conventional supervised learning
paradigm. However, this is not the case for the pretraining stage of LLMs. In
this phase, LLMs use self-supervised learning, where the model generates its
own labels from the input data.
This first training stage of an LLM is also known as pretraining, creating an initial pre-
trained LLM, often called a base or foundation model. A typical example of such a model
is the GPT-3 model. After obtaining a pretrained LLM from training on large text datasets, where the LLM is trained to predict the next word in the text, we can further train the LLM on
labeled data, also known as fine-tuning.
The two most popular categories of fine-tuning LLMs are instruction fine-tuning and
classification fine-tuning. In instruction fine-tuning, the labeled dataset consists of
instruction and answer pairs, such as a query to translate a text accompanied by the
correctly translated text. In classification fine-tuning, the labeled dataset consists of
texts and associated class labels—for example, emails associated with “spam” and “not
spam” labels.

## Introducing the transformer architecture
Most modern LLMs rely on the transformer architecture, which is a deep neural net-
work architecture introduced in the 2017 paper “Attention Is All You Need” (https://
arxiv.org/abs/1706.03762). To understand LLMs, we must understand the original
transformer, which was developed for machine translation, translating English texts to
German and French. A simplified version of the transformer architecture is depicted
in figure 1.4.
The transformer architecture consists of two submodules: an encoder and a
decoder. The encoder module processes the input text and encodes it into a series of
numerical representations or vectors that capture the contextual information of the
input. Then, the decoder module takes these encoded vectors and generates the out-
put text. In a translation task, for example, the encoder would encode the text from
the source language into vectors, and the decoder would decode these vectors to gen-
erate text in the target language. Both the encoder and decoder consist of many layers
connected by a so-called self-attention mechanism.
A key component of transformers and LLMs is the self-attention mechanism (not
shown), which allows the model to weigh the importance of different words or tokens
in a sequence relative to each other. This mechanism enables the model to capture
long-range dependencies and contextual relationships within the input data, enhanc-
ing its ability to generate coherent and contextually relevant output.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image3.png?raw=true)

BERT, which is built upon the original transformer’s encoder submodule, differs
in its training approach from GPT. While GPT is designed for generative tasks, BERT
and its variants specialize in masked word prediction, where the model predicts masked
or hidden words in a given sentence, as shown in figure below. This unique training strategy
equips BERT with strengths in text classification tasks, including sentiment prediction
and document categorization. As an application of its capabilities, as of this writing, X
(formerly Twitter) uses BERT to detect toxic content.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image4.png?raw=true)

GPT, on the other hand, focuses on the decoder portion of the original transformer
architecture and is designed for tasks that require generating texts. This includes
machine translation, text summarization, fiction writing, writing computer code,
and more.
GPT models, primarily designed and trained to perform text completion tasks,
also show remarkable versatility in their capabilities. These models are adept at exe-
cuting both zero-shot and few-shot learning tasks. Zero-shot learning refers to the abil-
ity to generalize to completely unseen tasks without any prior specific examples. On
the other hand, few-shot learning involves learning from a minimal number of exam-
ples the user provides as input, as shown in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image5.png?raw=true)

**Transformers vs. LLMs**
Today’s LLMs are based on the transformer architecture. Hence, transformers and
LLMs are terms that are often used synonymously in the literature. However, note
that not all transformers are LLMs since transformers can also be used for com-
puter vision. Also, not all LLMs are transformers, as there are LLMs based on recur-
rent and convolutional architectures. The main motivation behind these alternative
approaches is to improve the computational efficiency of LLMs. Whether these alter-
native LLM architectures can compete with the capabilities of transformer-based
LLMs and whether they are going to be adopted in practice remains to be seen. For
simplicity, I use the term “LLM” to refer to transformer-based LLMs similar to GPT.


## Utilizing large datasets
The large training datasets for popular GPT- and BERT-like models represent diverse
and comprehensive text corpora encompassing billions of words, which include a vast
array of topics and natural and computer languages.
The pretrained nature of these models makes them incredibly versatile for further
fine-tuning on downstream tasks, which is why they are also known as base or founda-
tion models. Pretraining LLMs requires access to significant resources and is very
expensive. For example, the GPT-3 pretraining cost is estimated to be $4.6 million in
terms of cloud computing credits (https://mng.bz/VxEW).
The good news is that many pretrained LLMs, available as open source models,
can be used as general-purpose tools to write, extract, and edit texts that were not
part of the training data. Also, LLMs can be fine-tuned on specific tasks with rela-
tively smaller datasets, reducing the computational resources needed and improving
performance.

## A closer look at the GPT architecture
GPT was originally introduced in the paper “Improving Language Understanding by
Generative Pre-Training” (https://mng.bz/x2qg) by Radford et al. from OpenAI.
GPT-3 is a scaled-up version of this model that has more parameters and was trained
on a larger dataset. In addition, the original model offered in ChatGPT was created by
fine-tuning GPT-3 on a large instruction dataset using a method from OpenAI’s
InstructGPT paper (https://arxiv.org/abs/2203.02155).

The model is simply trained to predict the next **word**
In the next-word prediction pretraining task for GPT models, the system learns to predict the upcoming word in a sentence by looking at the words that have come before it. This
approach helps the model understand how words and phrases typically fit together in language, forming a foundation that can be applied to various other tasks.
The next-word prediction task is a form of self-supervised learning, which is a form of
self-labeling. This means that we don’t need to collect labels for the training data
explicitly but can use the structure of the data itself: we can use the next word in a sen-
tence or document as the label that the model is supposed to predict. Since this next-
word prediction task allows us to create labels “on the fly,” it is possible to use massive
unlabeled text datasets to train LLMs.

Compared to the original transformer architecture, the
general GPT architecture is relatively simple. Essentially, it’s just the decoder part
without the encoder (figure below). Since decoder-style models like GPT generate text
by predicting text one word at a time, they are considered a type of autoregressive
model. Autoregressive models incorporate their previous outputs as inputs for future

![alt text](https://github.com/Rezashatery/LLM/blob/main/image6.png?raw=true)

Although the original transformer model, consisting of encoder and decoder blocks,
was explicitly designed for language translation, GPT models—despite their larger yet simpler decoder-only architecture aimed at next-word prediction—are also capable of
performing translation tasks. This capability was initially unexpected to researchers, as
it emerged from a model primarily trained on a next-word prediction task, which is a
task that did not specifically target translation.
The ability to perform tasks that the model wasn’t explicitly trained to perform is
called an **emergent behavior**. This capability isn’t explicitly taught during training but
emerges as a natural consequence of the model’s exposure to vast quantities of multi-
lingual data in diverse contexts. The fact that GPT models can “learn” the translation
patterns between languages and perform translation tasks even though they weren’t
specifically trained for it demonstrates the benefits and capabilities of these large-
scale, generative language models. We can perform diverse tasks without using diverse
models for each.


## Building a large language model
Now that we’ve laid the groundwork for understanding LLMs, let’s code one from
scratch. We will take the fundamental idea behind GPT as a blueprint and tackle this
in three stages, as outlined in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image7.png?raw=true)

The three main stages of coding an LLM are implementing the LLM architecture and data preparation process (stage 1), pretraining an LLM to create a foundation model (stage 2), and fine-tuning the foundation model to become a personal assistant or text classifier (stage 3).



# Working with text data (CHAPTER 2)
During the pretraining stage, LLMs process text one word at a time. Training
LLMs with millions to billions of parameters using a next-word prediction task
yields models with impressive capabilities. These models can then be further fine-
tuned to follow general instructions or perform specific target tasks. But before we
can implement and train LLMs, we need to prepare the training dataset, as illus-
trated in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image8.png?raw=true)

This involves splitting text
into individual word and subword tokens, which can then be encoded into vector rep-
resentations for the LLM. You’ll also learn about advanced tokenization schemes like
byte pair encoding, which is utilized in popular LLMs like GPT. Lastly, we’ll imple-
ment a sampling and data-loading strategy to produce the input-output pairs neces-
sary for training LLMs.

## Understanding word embeddings

Deep neural network models, including LLMs, cannot process raw text directly. Since
text is categorical, it isn’t compatible with the mathematical operations used to imple-
ment and train neural networks. Therefore, we need a way to represent words as
continuous-valued vectors.

The concept of converting data into a vector format is often referred to as **embedding**.
Using a specific neural network layer or another pretrained neural network model, we
can embed different data types—for example, video, audio, and text, as illustrated in
figure below. However, it’s important to note that different data formats require distinct
embedding models. For example, an embedding model designed for text would not
be suitable for embedding audio or video data.the primary purpose of embeddings is to convert nonnumeric data into a format that neural networks can process.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image9.png?raw=true)

Several algorithms and frameworks have been developed to generate word embed-
dings. One of the earlier and most popular examples is the **Word2Vec** approach.
Word2Vec trained neural network architecture to generate word embeddings by predicting the context of a word given the target word or vice versa. The main idea
behind Word2Vec is that words that appear in similar contexts tend to have similar
meanings. Consequently, when projected into two-dimensional word embeddings for
visualization purposes, similar terms are clustered together, as shown in figure below.
Word embeddings can have varying dimensions, from one to thousands. A higher
dimensionality might capture more nuanced relationships but at the cost of computa-
tional efficiency.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image10.png?raw=true)

If word embeddings are two-dimensional, we can plot them in a two-
dimensional scatterplot for visualization purposes as shown here. When using word
embedding techniques, such as Word2Vec, words corresponding to similar concepts
often appear close to each other in the embedding space. For instance, different types
of birds appear closer to each other in the embedding space than in countries and cities.
While we can use pretrained models such as Word2Vec to generate embeddings for
machine learning models, LLMs commonly produce their own embeddings that are
part of the input layer and are updated during training. The advantage of optimizing
the embeddings as part of the LLM training instead of using Word2Vec is that the
embeddings are optimized to the specific task and data at hand.
When working with LLMs, we typically use embeddings with a much higher dimensionality. For
both GPT-2 and GPT-3, the embedding size (often referred to as the dimensionality
of the model’s hidden states) varies based on the specific model variant and size. It
is a tradeoff between performance and efficiency. The smallest GPT-2 models (117M
and 125M parameters) use an embedding size of 768 dimensions to provide con-
crete examples.



## Tokenizing text
Let’s discuss how we split input text into individual tokens, a required preprocessing
step for creating embeddings for an LLM. These tokens are either individual words or
special characters, including punctuation characters, as shown in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image11.png?raw=true)

The text we will tokenize for LLM training is “The Verdict,” a short story by Edith
Wharton, which has been released into the public domain and is thus permitted to be
used for LLM training tasks. The text is available on Wikisource at https://en.wikisource
.org/wiki/The_Verdict, and you can copy and paste it into a text file, which I copied
into a text file "the-verdict.txt" in this repository.

```python
import urllib.request
url = ("https://raw.githubusercontent.com/rasbt/"
"LLMs-from-scratch/main/ch02/01_main-chapter-code/"
"the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)
```
Next, we can load the the-verdict.txt file using Python’s standard file reading utilities.
```python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])
```

The print command prints the total number of characters followed by the first 100
characters of this file for illustration purposes:
```python
Total number of character: 20479
I HAD always thought Jack Gisburn rather a cheap genius-though a good fellow
enough-so it was no
```

Our goal is to tokenize this 20,479-character short story into individual words and spe-
cial characters that we can then turn into embeddings for LLM training.

**NOTE**
It’s common to process millions of articles and hundreds of thousands
of books—many gigabytes of text—when working with LLMs. However, for
educational purposes, it’s sufficient to work with smaller text samples like a
single book to illustrate the main ideas behind the text processing steps and
to make it possible to run it in a reasonable time on consumer hardware.

How can we best split this text to obtain a list of tokens? For this, we go on a small
excursion and use Python’s regular expression library re for illustration purposes.
```python
import re
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)
```

Result: 
['Hello,', ' ', 'world.', ' ', 'This,', ' ', 'is', ' ', 'a', ' ', 'test.']

This simple tokenization scheme mostly works for separating the example text into
individual words; however, some words are still connected to punctuation characters
that we want to have as separate list entries.
We also refrain from making all text lowercase because capitalization helps LLMs distinguish between proper nouns and common nouns, understand sentence structure, and learn to generate text with proper capitalization.

Let’s modify the regular expression splits on whitespaces (\s), commas, and peri-
ods ([,.]):
```python
result = re.split(r'([,.]|\s)', text)
print(result)
```
Result: 
['Hello', ',', '', ' ', 'world', '.', '', ' ', 'This', ',', '', ' ', 'is',
' ', 'a', ' ', 'test', '.', '']

A small remaining problem is that the list still includes whitespace characters. Option-
ally, we can remove these redundant characters safely as follows:

```python
result = [item for item in result if item.strip()]
print(result)
```
Result: ['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']

**NOTE**
When developing a simple tokenizer, whether we should encode
whitespaces as separate characters or just remove them depends on our appli-
cation and its requirements. Removing whitespaces reduces the memory and
computing requirements. However, keeping whitespaces can be useful if we
train models that are sensitive to the exact structure of the text (for example,
Python code, which is sensitive to indentation and spacing). Here, we remove
whitespaces for simplicity and brevity of the tokenized outputs. Later, we will
switch to a tokenization scheme that includes whitespaces.

Let’s
modify it a bit further so that it can also handle other types of punctuation, such as ques-
tion marks, quotation marks, and the double-dashes we have seen earlier in the first 100
characters of Edith Wharton’s short story, along with additional special characters:
```python
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)
```
Result: ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']

![alt text](https://github.com/Rezashatery/LLM/blob/main/image12.png?raw=true)

Now that we have a basic tokenizer working, let’s apply it to Edith Wharton’s entire
short story:
```python
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
```
first 30 tokens for a quick visual check:
['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a',
'cheap', 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough',
'--', 'so', 'it', 'was', 'no', 'great', 'surprise', 'to', 'me', 'to',
'hear', 'that', ',', 'in']

## Converting tokens into token IDs
Next, let’s convert these tokens from a Python string to an integer representation to
produce the token IDs. This conversion is an intermediate step before converting the
token IDs into embedding vectors.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image13.png?raw=true)

We build a vocabulary by tokenizing the entire text in a training dataset into individual
tokens. These individual tokens are then sorted alphabetically, and duplicate tokens are removed. The unique tokens are then aggregated into a vocabulary that defines a mapping from each unique token to a unique integer value. The depicted vocabulary is purposefully small and contains no punctuation or special characters for simplicity.

Now that we have tokenized Edith Wharton’s short story and assigned it to a Python
variable called preprocessed, let’s create a list of all unique tokens and sort them
alphabetically to determine the vocabulary size:
```python
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)
```
After determining that the vocabulary size is 1,130 via this code, we create the vocabu-
lary and print its first 51 entries for illustration purposes.

```python
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
print(item)
if i >= 50:
break
```
The output is
('!', 0)
('"', 1)
("'", 2)
...
('Her', 49)
('Hermia', 50)

As we can see, the dictionary contains individual tokens associated with unique inte-
ger labels. Our next goal is to apply this vocabulary to convert new text into token IDs.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image14.png?raw=true)

Starting with a new text sample, we tokenize the text and use the vocabulary to convert
the text tokens into token IDs. The vocabulary is built from the entire training set and can be applied to the training set itself and any new text samples. The depicted vocabulary contains no punctuation or special characters for simplicity.
When we want to convert the outputs of an LLM from numbers back into text, we need a
way to turn token IDs into text. For this, we can create an inverse version of the vocabu-
lary that maps token IDs back to the corresponding text tokens.

Let’s implement a complete tokenizer class in Python with an encode method that
splits text into tokens and carries out the string-to-integer mapping to produce token
IDs via the vocabulary. In addition, we’ll implement a decode method that carries out
the reverse integer-to-string mapping to convert the token IDs back into text. The fol-
lowing listing shows the code for this tokenizer implementation.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image15.png?raw=true)

Using the SimpleTokenizerV1 Python class, we can now instantiate new tokenizer
objects via an existing vocabulary, which we can then use to encode and decode text,
as illustrated in figure below.
![alt text](https://github.com/Rezashatery/LLM/blob/main/image16.png?raw=true)

Let’s instantiate a new tokenizer object from the SimpleTokenizerV1 class and
tokenize a passage from Edith Wharton’s short story to try it out in practice:
```python
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
```

So far, so good. We implemented a tokenizer capable of tokenizing and detokeniz-
ing text based on a snippet from the training set. Let’s now apply it to a new text sam-
ple not contained in the training set:
```python
text = "Hello, do you like tea?"
print(tokenizer.encode(text))
```
Executing this code will result in the following error:
KeyError: 'Hello'

The problem is that the word “Hello” was not used in the “The Verdict” short story.
Hence, it is not contained in the vocabulary. This highlights the need to consider
large and diverse training sets to extend the vocabulary when working on LLMs.

## Adding special context tokens
We need to modify the tokenizer to handle unknown words. We also need to address
the usage and addition of special context tokens that can enhance a model’s under-
standing of context or other relevant information in the text. These special tokens
can include markers for unknown words and document boundaries, for example. In
particular, we will modify the vocabulary and tokenizer, SimpleTokenizerV2, to sup-
port two new tokens, <|unk|> and <|endoftext|>, as illustrated in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image17.png?raw=true)

We add special tokens to a vocabulary to deal with certain contexts. For instance,
we add an <|unk|> token to represent new and unknown words that were not part of the training
data and thus not part of the existing vocabulary. Furthermore, we add an <|endoftext|>
token that we can use to separate two unrelated text sources.
we add a token between unrelated texts.For example, when training GPT-like LLMs on multiple independent documents or books, it is common to insert a token before each document or book that follows a previous text source, as illustrated in figure below. This helps the LLM understand that although these text sources are concatenated for training, they are, in fact,
unrelated.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image18.png?raw=true)

When working with multiple independent text source, we add <|endoftext|>
tokens between these texts. These <|endoftext|> tokens act as markers, signaling the
start or end of a particular segment, allowing for more effective processing and understanding
by the LLM.

Let’s now modify the vocabulary to include these two special tokens, <unk> and
<|endoftext|>, by adding them to our list of all unique words:
```python
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))
```
Based on the code output, we can confirm that the two new special tokens were
indeed successfully incorporated into the vocabulary. Next, we adjust the tokenizer
from previous code accordingly as shown in the following listing.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image19.png?raw=true)

Let’s now try this new tokenizer out in practice. For this, we will use a simple text
sample that we concatenate from two independent and unrelated sentences:
```python
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)
```
The output:
Hello, do you like tea? <|endoftext|> In the sunlit terraces of
the palace.

Next, let’s tokenize the sample text using the SimpleTokenizerV2 on the vocab we
previously created.

```python
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
```
Result: [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]

Depending on the LLM, some researchers also consider additional special tokens
such as the following:

- [BOS] (beginning of sequence)—This token marks the start of a text. It signifies to
the LLM where a piece of content begins.
- [EOS] (end of sequence)—This token is positioned at the end of a text and
is especially useful when concatenating multiple unrelated texts, similar to
<|endoftext|>. For instance, when combining two different Wikipedia arti-
cles or books, the [EOS] token indicates where one ends and the next begins.
- [PAD] (padding)—When training LLMs with batch sizes larger than one, the
batch might contain texts of varying lengths. To ensure all texts have the same
length, the shorter texts are extended or “padded” using the [PAD] token, up to
the length of the longest text in the batch.

The tokenizer used for GPT models does not need any of these tokens; it only uses an
<|endoftext|> token for simplicity. <|endoftext|> is analogous to the [EOS] token.
<|endoftext|> is also used for padding.
Moreover, the tokenizer used for GPT models also doesn’t use an <|unk|> token
for out-of-vocabulary words. Instead, GPT models use a byte pair encoding tokenizer,
which breaks words down into subword units.

## Byte pair encoding
Let’s look at a more sophisticated tokenization scheme based on a concept called byte
pair encoding (BPE). The BPE tokenizer was used to train LLMs such as GPT-2, GPT-3,
and the original model used in ChatGPT.
Since implementing BPE can be relatively complicated, we will use an existing
Python open source library called tiktoken.

The code we will use is based on tiktoken 0.7.0. 
Once installed, we can instantiate the BPE tokenizer from tiktoken as follows:
```python
from importlib.metadata import version
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
```

The usage of this tokenizer is similar to the SimpleTokenizerV2 we implemented pre-
viously via an encode method
```python
text = (
"Hello, do you like tea? <|endoftext|> In the sunlit terraces"
"of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
```
We can then convert the token IDs back into text using the decode method, similar to
our SimpleTokenizerV2:
```python
strings = tokenizer.decode(integers)
print(strings)
```
We can make two noteworthy observations based on the token IDs and decoded text.
First, the <|endoftext|> token is assigned a relatively large token ID, namely, 50256.
In fact, the BPE tokenizer, which was used to train models such as GPT-2, GPT-3, and
the original model used in ChatGPT, has a total vocabulary size of 50,257, with
<|endoftext|> being assigned the largest token ID.

Second, the BPE tokenizer encodes and decodes unknown words, such as
someunknownPlace, correctly. The BPE tokenizer can handle any unknown word. How
does it achieve this without using <|unk|> tokens?
The algorithm underlying BPE breaks down words that aren’t in its predefined
vocabulary into smaller subword units or even individual characters, enabling it to
handle out-of-vocabulary words. So, thanks to the BPE algorithm, if the tokenizer
encounters an unfamiliar word during tokenization, it can represent it as a sequence
of subword tokens or characters, as illustrated in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image20.png?raw=true)
The ability to break down unknown words into individual characters ensures that
the tokenizer and, consequently, the LLM that is trained with it can process any text,
even if it contains words that were not present in its training data.
in short, BPE builds its vocabulary by iteratively merging frequent characters into sub-
words and frequent subwords into words. For example, BPE starts with adding all indi-
vidual single characters to its vocabulary (“a,” “b,” etc.). In the next stage, it merges
character combinations that frequently occur together into subwords. For example,
“d” and “e” may be merged into the subword “de,” which is common in many English words like “define,” “depend,” “made,” and “hidden.” The merges are determined by a frequency cutoff.

## Data sampling with a sliding window

The next step in creating the embeddings for the LLM is to generate the input–target
pairs required for training an LLM. What do these input–target pairs look like? As we
already learned, LLMs are pretrained by predicting the next word in a text, as depicted
in figure below.
![alt text](https://github.com/Rezashatery/LLM/blob/main/image21.png?raw=true)

Given a text sample, extract input blocks as subsamples that serve as
input to the LLM, and the LLM’s prediction task during training is to predict the next
word that follows the input block. During training, we mask out all words that are past
the target. Note that the text shown in this figure must undergo tokenization before
the LLM can process it; however, this figure omits the tokenization step for clarity.

Let’s implement a data loader that fetches the input–target pairs in figure above from
the training dataset using a sliding window approach. To get started, we will tokenize
the whole “The Verdict” short story using the BPE tokenizer:
```python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
```
Executing this code will return 5145, the total number of tokens in the training set,
after applying the BPE tokenizer.
Next, we remove the first 50 tokens from the dataset for demonstration purposes,
as it results in a slightly more interesting text passage in the next steps:
```python
enc_sample = enc_text[50:]
```
One of the easiest and most intuitive ways to create the input–target pairs for the next-
word prediction task is to create two variables, x and y, where x contains the input
tokens and y contains the targets, which are the inputs shifted by 1:

![alt text](https://github.com/Rezashatery/LLM/blob/main/image22.png?raw=true)

Running the previous code prints the following output:
x: [290, 4920, 2241, 287]
y:       [4920, 2241, 287, 257]

By processing the inputs along with the targets, which are the inputs shifted by one
position, we can create the next-word prediction tasks as follows:
```python
for i in range(1, context_size+1):
context = enc_sample[:i]
desired = enc_sample[i]
print(context, "---->", desired)
```
The code prints
```python
[290] ----> 4920
[290, 4920] ----> 2241
[290, 4920, 2241] ----> 287
[290, 4920, 2241, 287] ----> 257
```
Everything left of the arrow (---->) refers to the input an LLM would receive, and
the token ID on the right side of the arrow represents the target token ID that the
LLM is supposed to predict.
Let’s repeat the previous code but convert the token IDs
into text:
```python
for i in range(1, context_size+1):
context = enc_sample[:i]
desired = enc_sample[i]
print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
```
The following outputs show how the input and outputs look in text format:
```python
and ----> established
and established ----> himself
and established himself ----> in
and established himself in ----> a
```
We’ve now created the input–target pairs that we can use for LLM training.
There’s only one more task before we can turn the tokens into embeddings: imple-
menting an efficient data loader that iterates over the input dataset and returns the inputs and targets as PyTorch tensors, which can be thought of as multidimensional
arrays. In particular, we are interested in returning two tensors: an input tensor con-
taining the text that the LLM sees and a target tensor that includes the targets for the
LLM to predict, as depicted in figure below. While the figure shows the tokens in string
format for illustration purposes, the code implementation will operate on token IDs
directly since the encode method of the BPE tokenizer performs both tokenization
and conversion into token IDs as a single step
![alt text](https://github.com/Rezashatery/LLM/blob/main/image23.png?raw=true)

To implement efficient data loaders, we collect the inputs in a tensor, x, where each row
represents one input context. A second tensor, y, contains the corresponding prediction targets (next words), which are created by shifting the input by one position.

```python
import torch
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
def __init__(self, txt, tokenizer, max_length, stride):
    self.input_ids = []
    self.target_ids = []
    
    token_ids = tokenizer.encode(txt)
for i in range(0, len(token_ids) - max_length, stride): #Uses a sliding window to chunk the book into overlapping sequences of max_length
    input_chunk = token_ids[i:i + max_length]
    target_chunk = token_ids[i + 1: i + max_length + 1]
    self.input_ids.append(torch.tensor(input_chunk))
    self.target_ids.append(torch.tensor(target_chunk))
def __len__(self):
    return len(self.input_ids)  #Returns the total number of rows in the dataset
def __getitem__(self, idx): 
    return self.input_ids[idx], self.target_ids[idx] #Returns a single row from the dataset
```
The GPTDatasetV1 class is based on the PyTorch Dataset class and defines how indi-
vidual rows are fetched from the dataset, where each row consists of a number of
token IDs (based on a max_length) assigned to an input_chunk tensor. The target_
chunk tensor contains the corresponding targets. I recommend reading on to see what
the data returned from this dataset looks like when we combine the dataset with a
PyTorch DataLoader—this will bring additional intuition and clarity.
The following code uses the GPTDatasetV1 to load the inputs in batches via a PyTorch
DataLoader.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image24.png?raw=true)

Let’s test the dataloader with a batch size of 1 for an LLM with a context size of 4 to
develop an intuition of how the GPTDatasetV1 class and the create_
dataloader_v1 function work together:

```python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
raw_text = f.read()
dataloader = create_dataloader_v1(
raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
```
Executing the preceding code prints the following:
[tensor([[40, 367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]
The first_batch variable contains two tensors: the first tensor stores the input token
IDs, and the second tensor stores the target token IDs. Since the max_length is set to
4, each of the two tensors contains four token IDs. Note that an input size of 4 is quite
small and only chosen for simplicity. It is common to train LLMs with input sizes of at
least 256.
To understand the meaning of stride=1, let’s fetch another batch from this dataset:
```python
second_batch = next(data_iter)
print(second_batch)
```
The second batch has the following contents:
[tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]

If we compare the first and second batches, we can see that the second batch’s token
IDs are shifted by one position (for example, the second ID in the first batch’s input is
367, which is the first ID of the second batch’s input). The stride setting dictates the
number of positions the inputs shift across batches, emulating a sliding window
approach, as demonstrated in figure below.
Batch sizes of 1, such as we have sampled from the data loader so far, are useful for
illustration purposes. If you have previous experience with deep learning, you may
know that small batch sizes require less memory during training but lead to more noisy model updates. Just like in regular deep learning, the batch size is a tradeoff and
a hyperparameter to experiment with when training LLMs.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image25.png?raw=true)


## Creating token embeddings

The last step in preparing the input text for LLM training is to convert the token IDs
into embedding vectors, as shown in figure 2.15. As a preliminary step, we must initialize these embedding weights with random values. This initialization serves as the starting
point for the LLM’s learning process.A continuous vector representation, or embedding, is necessary since GPT-like LLMs are deep neural networks trained with the backpropagation algorithm.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image26.png?raw=true)

Let’s see how the token ID to embedding vector conversion works with a hands-on
example. Suppose we have the following four input tokens with IDs 2, 3, 5, and 1:

input_ids = torch.tensor([2, 3, 5, 1])
For the sake of simplicity, suppose we have a small vocabulary of only 6 words (instead
of the 50,257 words in the BPE tokenizer vocabulary), and we want to create embed-
dings of size 3 (in GPT-3, the embedding size is 12,288 dimensions):    
vocab_size = 6
output_dim = 3

Using the vocab_size and output_dim, we can instantiate an embedding layer in
PyTorch, setting the random seed to 123 for reproducibility purposes:

```python
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
```
The print statement prints the embedding layer’s underlying weight matrix:
```python
Parameter containing:
tensor([[ 0.3374, -0.1778, -0.1690],
[ 0.9178, 1.5810, 1.3010],
[ 1.2753, -0.2010, -0.1606],
[-0.4015, 0.9666, -1.1481],
[-1.1589, 0.3255, -0.6315],
[-2.8400, -0.7849, -1.4096]], requires_grad=True)
```
The weight matrix of the embedding layer contains small, random values. These val-
ues are optimized during LLM training as part of the LLM optimization itself. More-
over, we can see that the weight matrix has six rows and three columns. There is one row
for each of the six possible tokens in the vocabulary, and there is one column for each of
the three embedding dimensions.

**NOTE** For those who are familiar with one-hot encoding, the embedding
layer approach described here is essentially just a more efficient way of imple-
menting one-hot encoding followed by matrix multiplication in a fully con-
nected layer, which is illustrated in the supplementary code on GitHub at
https://mng.bz/ZEB5. Because the embedding layer is just a more efficient
implementation equivalent to the one-hot encoding and matrix-multiplica-
tion approach, it can be seen as a neural network layer that can be optimized
via backpropagation.

Each row in this output matrix is obtained via a lookup operation from the embed-
ding weight matrix, as illustrated in figure below.
Having now created embedding vectors from token IDs, next we’ll add a small
modification to these embedding vectors to encode positional information about a
token within a text.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image27.png?raw=true)

## Encoding word positions
In principle, token embeddings are a suitable input for an LLM. However, a minor
shortcoming of LLMs is that their self-attention mechanism doesn’t
have a notion of position or order for the tokens within a sequence. The way the pre-
viously introduced embedding layer works is that the same token ID always gets
mapped to the same vector representation, regardless of where the token ID is posi-
tioned in the input sequence, as shown in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image28.png?raw=true)

The embedding layer converts a token ID into the same vector
representation regardless of where it is located in the input sequence. For
example, the token ID 5, whether it’s in the first or fourth position in the
token ID input vector, will result in the same embedding vector.
In principle, the deterministic, position-independent embedding of the token ID is
good for reproducibility purposes. However, since the self-attention mechanism of
LLMs itself is also position-agnostic, it is helpful to inject additional position informa-
tion into the LLM.
To achieve this, we can use two broad categories of position-aware embeddings: rela-
tive positional embeddings and absolute positional embeddings. Absolute positional
embeddings are directly associated with specific positions in a sequence. For each posi-
tion in the input sequence, a unique embedding is added to the token’s embedding to
convey its exact location. For instance, the first token will have a specific positional
embedding, the second token another distinct embedding, and so on, as illustrated in
figure below.
![alt text](https://github.com/Rezashatery/LLM/blob/main/image29.png?raw=true)

Positional embeddings are added to the token embedding vector to create the
input embeddings for an LLM. The positional vectors have the same dimension as the original
token embeddings. The token embeddings are shown with value 1 for simplicity.
Instead of focusing on the absolute position of a token, the emphasis of relative posi-
tional embeddings is on the relative position or distance between tokens. This means
the model learns the relationships in terms of “how far apart” rather than “at which
exact position.” The advantage here is that the model can generalize better to sequences
of varying lengths, even if it hasn’t seen such lengths during training.Both types of positional embeddings aim to augment the capacity of LLMs to understand the order and relationships between tokens, ensuring more accurate and context-aware predictions. The choice between them often depends on the specific application and the nature of the data being processed.
OpenAI’s GPT models use absolute positional embeddings that are optimized
during the training process rather than being fixed or predefined like the positional
encodings in the original transformer model. This optimization process is part of the
model training itself. For now, let’s create the initial positional embeddings to create the
LLM inputs.
Previously, we focused on very small embedding sizes for simplicity. Now, let’s con-
sider more realistic and useful embedding sizes and encode the input tokens into a
256-dimensional vector representation, which is smaller than what the original GPT-3
model used (in GPT-3, the embedding size is 12,288 dimensions) but still reasonable
for experimentation. Furthermore, we assume that the token IDs were created by the
BPE tokenizer we implemented earlier, which has a vocabulary size of 50,257:

```python
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
```
Using the previous token_embedding_layer, if we sample data from the data loader,
we embed each token in each batch into a 256-dimensional vector. If we have a batch
size of 8 with four tokens each, the result will be an 8 × 4 × 256 tensor.
Let’s instantiate the data loader first:
```python
max_length = 4
dataloader = create_dataloader_v1(
raw_text, batch_size=8, max_length=max_length,
stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)
```
This code prints
```python
Token IDs:
tensor([[40,367, 2885, 1464],
[ 1807, 3619,402,271],
[10899, 2138,257, 7026],
[15632,438, 2016,257],
[ 922, 5891, 1576,438],
[ 568,340,373,645],
[ 1049, 5975,284,502],
[ 284, 3285,326,11]])
Inputs shape:
torch.Size([8, 4])
```
As we can see, the token ID tensor is 8 × 4 dimensional, meaning that the data batch
consists of eight text samples with four tokens each.Let’s now use the embedding layer to embed these token IDs into 256-dimensional vectors:
```python
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
```
The print function call returns:
torch.Size([8, 4, 256])

The 8 × 4 × 256–dimensional tensor output shows that each token ID is now embed-
ded as a 256-dimensional vector.
For a GPT model’s absolute embedding approach, we just need to create another
embedding layer that has the same embedding dimension as the token_embedding_
layer:
```python
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)
```
The input to the pos_embeddings is usually a placeholder vector torch.arange(con-
text_length), which contains a sequence of numbers 0, 1, ..., up to the maximum
input length –1. The context_length is a variable that represents the supported input
size of the LLM. Here, we choose it similar to the maximum length of the input text.
In practice, input text can be longer than the supported context length, in which case
we have to truncate the text.
The output of the print statement is
torch.Size([4, 256])

As we can see, the positional embedding tensor consists of four 256-dimensional vec-
tors. We can now add these directly to the token embeddings, where PyTorch will add
the 4 × 256–dimensional pos_embeddings tensor to each 4 × 256–dimensional token
embedding tensor in each of the eight batches:
```python
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
```
The print output is
torch.Size([8, 4, 256])
The input_embeddings we created, as summarized in figure below, are the embedded
input examples that can now be processed by the main LLM modules,

![alt text](https://github.com/Rezashatery/LLM/blob/main/image30.png?raw=true)

# Coding attention mechanisms (CHAPTER 3)
At this point, you know how to prepare the input text for training LLMs by splitting
text into individual word and subword tokens, which can be encoded into vector rep-
resentations, embeddings, for the LLM.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image31.png?raw=true)
parts of the LLM surrounding the self-attention mechanism to see it in action and to
create a model to generate text.
We will implement four different variants of attention mechanisms, as illustrated in
figure below. These different attention variants build on each other, and the goal is to arrive at a compact and efficient implementation of multi-head attention that we can
then plug into the LLM architecture.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image32.png?raw=true)

## The problem with modeling long sequences

Before we dive into the self-attention mechanism at the heart of LLMs, let’s consider
the problem with pre-LLM architectures that do not include attention mechanisms.
Suppose we want to develop a language translation model that translates text from
one language into another. As shown in figure 3.3, we can’t simply translate a text word
by word due to the grammatical structures in the source and target language.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image33.png?raw=true)

To address this problem, it is common to use a deep neural network with two submod-
ules, an encoder and a decoder. The job of the encoder is to first read in and process the
entire text, and the decoder then produces the translated text.
Before the advent of transformers, recurrent neural networks (RNNs) were the most
popular encoder–decoder architecture for language translation. An RNN is a type of
neural network where outputs from previous steps are fed as inputs to the current step, making them well-suited for sequential data like text.

In an encoder–decoder RNN, the input text is fed into the encoder, which pro-
cesses it sequentially. The encoder updates its hidden state (the internal values at the
hidden layers) at each step, trying to capture the entire meaning of the input sen-
tence in the final hidden state, as illustrated in figure below. The decoder then takes this
final hidden state to start generating the translated sentence, one word at a time. It
also updates its hidden state at each step, which is supposed to carry the context nec-
essary for the next-word prediction.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image34.png?raw=true)


While we don’t need to know the inner workings of these encoder–decoder RNNs,
the key idea here is that the encoder part processes the entire input text into a hid-
den state (memory cell). The decoder then takes in this hidden state to produce the
output. You can think of this hidden state as an embedding vector.
The big limitation of encoder–decoder RNNs is that the RNN can’t directly access
earlier hidden states from the encoder during the decoding phase. Consequently, it
relies solely on the current hidden state, which encapsulates all relevant information.
This can lead to a loss of context, especially in complex sentences where dependen-
cies might span long distances.

## Capturing data dependencies with attention mechanisms

Although RNNs work fine for translating short sentences, they don’t work well for lon-
ger texts as they don’t have direct access to previous words in the input. One major
shortcoming in this approach is that the RNN must remember the entire encoded
input in a single hidden state before passing it to the decoder.
Hence, researchers developed the Bahdanau attention mechanism for RNNs in
2014 which modifies the encoder–decoder RNN such that the decoder can
selectively access different parts of the input sequence at each decoding step as illus-
trated in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image35.png?raw=true)

Using an attention mechanism, the text-generating decoder part of the network can
access all input tokens selectively. This means that some input tokens are more important than others for generating a given output token. The importance is determined by the attention weights, which we will compute later.
consider the relevancy of, or “attend to,” all other positions in the same sequence
when computing the representation of a sequence. Self-attention is a key component
of contemporary LLMs based on the transformer architecture, such as the GPT series.
This chapter focuses on coding and understanding this self-attention mechanism
used in GPT-like models, as illustrated in figure 3.6. In the next chapter, we will code
the remaining parts of the LLM.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image36.png?raw=true)

Self-attention is a mechanism in transformers used to compute
more efficient input representations by allowing each position in a sequence to
interact with and weigh the importance of all other positions within the same
sequence.

## Attending to different parts of the input with self-attention
We’ll now cover the inner workings of the self-attention mechanism and learn how to
code it from the ground up. Self-attention serves as the cornerstone of every LLM
based on the transformer architecture. This topic may require a lot of focus and atten-
tion (no pun intended), but once you grasp its fundamentals, you will have con-
quered one of the toughest aspects of this book and LLM implementation in general.

**The “self” in self-attention**
In self-attention, the “self” refers to the mechanism’s ability to compute attention
weights by relating different positions within a single input sequence. It assesses and
learns the relationships and dependencies between various parts of the input itself,
such as words in a sentence or pixels in an image.
This is in contrast to traditional attention mechanisms, where the focus is on the rela-
tionships between elements of two different sequences, such as in sequence-to-
sequence models where the attention might be between an input sequence and an
output sequence.

## A simple self-attention mechanism without trainable weights
Let’s begin by implementing a simplified variant of self-attention, free from any train-
able weights, as summarized in figure below. The goal is to illustrate a few key concepts
in self-attention before adding trainable weights.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image37.png?raw=true)

The goal of self-attention is to compute a context vector for each input
element that combines information from all other input elements. In this example,
we compute the context vector z(2). The importance or contribution of each input
element for computing z(2) is determined by the attention weights 21 to 2T. When
computing z(2), the attention weights are calculated with respect to input element
x(2) and all other inputs.

Figure 3.7 shows an input sequence, denoted as x, consisting of T elements repre-
sented as x(1) to x(T). This sequence typically represents text, such as a sentence, that
has already been transformed into token embeddings.
For example, consider an input text like “Your journey starts with one step.” In this
case, each element of the sequence, such as x(1), corresponds to a d-dimensional
embedding vector representing a specific token, like “Your.” Figure above shows these
input vectors as three-dimensional embeddings.
In self-attention, our goal is to calculate context vectors z(i) for each element x(i)
in the input sequence. A context vector can be interpreted as an enriched embedding
vector.
To illustrate this concept, let’s focus on the embedding vector of the second input
element, x(2) (which corresponds to the token “journey”), and the corresponding con-
text vector, z(2), shown at the bottom of figure above. This enhanced context vector, z(2),
is an embedding that contains information about x(2) and all other input elements,
x(1) to x(T).
Context vectors play a crucial role in self-attention. Their purpose is to create
enriched representations of each element in an input sequence (like a sentence)
by incorporating information from all other elements in the sequence (figure above).
This is essential in LLMs, which need to understand the relationship and relevance
of words in a sentence to each other. Later, we will add trainable weights that help
an LLM learn to construct these context vectors so that they are relevant for the
LLM to generate the next token. But first, let’s implement a simplified self-atten-
tion mechanism to compute these weights and the resulting context vector one
step at a time.

Consider the following input sentence, which has already been embedded into
three-dimensional vectors. I’ve chosen a small embedding dimension
to ensure it fits on the page without line breaks:

```python
import torch
inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your
[0.55, 0.87, 0.66], # journey
[0.57, 0.85, 0.64], # starts
[0.22, 0.58, 0.33], # with
[0.77, 0.25, 0.10], # one
[0.05, 0.80, 0.55]] # step
)
```
The first step of implementing self-attention is to compute the intermediate values ω,
referred to as attention scores, as illustrated in figure below. Due to spatial constraints,
the figure displays the values of the preceding inputs tensor in a truncated version;
for example, 0.87 is truncated to 0.8. In this truncated version, the embeddings of the
words “journey” and “starts” may appear similar by random chance.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image38.png?raw=true)

The overall goal is to illustrate the computation of the context vector z(2) using the
second input element, x(2) as a query. This figure shows the first intermediate step, computing the attention scores w between the query x(2) and all other input elements as a dot product.

Figure above illustrates how we calculate the intermediate attention scores between the
query token and each input token. We determine these scores by computing the dot
product of the query, x(2), with every other input token:

```python
query = inputs[1]  #The second input token serves as the query.
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)
```

The computed attention scores are
tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

Beyond viewing the dot product operation as a mathematical tool that combines
two vectors to yield a scalar value, the dot product is a measure of similarity
because it quantifies how closely two vectors are aligned: a higher dot product indi-
cates a greater degree of alignment or similarity between the vectors. In the con-
text of self-attention mechanisms, the dot product determines the extent to which
each element in a sequence focuses on, or “attends to,” any other element: the
higher the dot product, the higher the similarity and attention score between two
elements.
In the next step, as shown in figure below, we normalize each of the attention scores we
computed previously. The main goal behind the normalization is to obtain attention
weights that sum up to 1. This normalization is a convention that is useful for interpre-
tation and maintaining training stability in an LLM. Here’s a straightforward method
for achieving this normalization step:
```python
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())
```

![alt text](https://github.com/Rezashatery/LLM/blob/main/image39.png?raw=true)

As the output shows, the attention weights now sum to 1:
Attention weights: tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
Sum: tensor(1.0000)
In practice, it’s more common and advisable to use the softmax function for normal-
ization. This approach is better at managing extreme values and offers more favorable 
gradient properties during training. The following is a basic implementation of the
softmax function for normalizing the attention scores:
```python
def softmax_naive(x):
return torch.exp(x) / torch.exp(x).sum(dim=0)
attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())
```
In addition, the softmax function ensures that the attention weights are always posi-
tive. This makes the output interpretable as probabilities or relative importance,
where higher weights indicate greater importance.Note that this naive softmax implementation (softmax_naive) may encounter numerical instability problems, such as overflow and underflow, when dealing with large or small input values. Therefore, in practice, it’s advisable to use the PyTorch implementation of softmax, which has been extensively optimized for performance:
```python
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())
```
In this case, it yields the same results as our previous softmax_naive function:
Attention weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
Sum: tensor(1.)
Now that we have computed the normalized attention weights, we are ready for the
final step, as shown in figure below: calculating the context vector z(2) by multiplying the
embedded input tokens, x(i), with the corresponding attention weights and then sum-
ming the resulting vectors. Thus, context vector z(2) is the weighted sum of all input vec-
tors, obtained by multiplying each input vector by its corresponding attention weight:
```python
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)
```
The results of this computation are
tensor([0.4419, 0.6515, 0.5683])

![alt text](https://github.com/Rezashatery/LLM/blob/main/image40.png?raw=true)
The final step, after calculating and normalizing the attention scores to obtain the
attention weights for query x(2), is to compute the context vector z(2). This context vector is a combination of all input vectors x(1) to x(T) weighted by the attention weights.

## Computing attention weights for all input tokens
So far, we have computed attention weights and the context vector for input 2, as
shown in the highlighted row in figure below. Now let’s extend this computation to cal-
culate attention weights and context vectors for all inputs.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image41.png?raw=true)

The highlighted row shows the attention weights for the second input element as a query. Now we will generalize the computation to obtain all other attention weights. (Please note that the numbers in this figure are truncated to two digits after the decimal point to reduce visual clutter. The values in each row should add up to 1.0 or 100%.)

We follow the same three steps as before (see figure below), except that we make a few
modifications in the code to compute all context vectors instead of only the second
one, z(2):
```python
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)
```
In step 1, we add an additional for loop to compute the dot products for all pairs of inputs.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image42.png?raw=true)

The resulting attention scores are as follows:
```python
tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
[0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
[0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
[0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
[0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
[0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])
```
Each element in the tensor represents an attention score between each pair of inputs,
as we saw before. Note that the values in that figure are normalized, which is
why they differ from the unnormalized attention scores in the preceding tensor. We
will take care of the normalization later.
When computing the preceding attention score tensor, we used for loops in
Python. However, for loops are generally slow, and we can achieve the same results
using matrix multiplication:
```python
attn_scores = inputs @ inputs.T
print(attn_scores)
```
We can visually confirm that the results are the same as before.
In step 2 of figure above, we normalize each row so that the values in each row sum to 1:
```python
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
```
In the context of using PyTorch, the dim parameter in functions like torch.softmax
specifies the dimension of the input tensor along which the function will be com-
puted. By setting dim=-1, we are instructing the softmax function to apply the nor-
malization along the last dimension of the attn_scores tensor. If attn_scores is a
two-dimensional tensor (for example, with a shape of [rows, columns]), it will nor-
malize across the columns so that the values in each row (summing over the column
dimension) sum up to 1.
We can verify that the rows indeed all sum to 1:
```python
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))
```
The result is
Row 2 sum: 1.0
All row sums: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])

In the third and final step of figure above, we use these attention weights to compute all
context vectors via matrix multiplication:
In the third and final step of figure above, we use these attention weights to compute all
context vectors via matrix multiplication:
```python
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
```
In the resulting output tensor, each row contains a three-dimensional context vector:
```python
tensor([[0.4421, 0.5931, 0.5790],
[0.4419, 0.6515, 0.5683],
[0.4431, 0.6496, 0.5671],
[0.4304, 0.6298, 0.5510],
[0.4671, 0.5910, 0.5266],
[0.4177, 0.6503, 0.5645]])
```
Next, we will add trainable weights, enabling the LLM to learn from data and improve its per-
formance on specific tasks.

## Implementing self-attention with trainable weights
Our next step will be to implement the self-attention mechanism used in the origi-
nal transformer architecture, the GPT models, and most other popular LLMs. This
self-attention mechanism is also called scaled dot-product attention. Figure below shows
how this self-attention mechanism fits into the broader context of implementing
an LLM.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image43.png?raw=true)

As illustrated in figure apove, the self-attention mechanism with trainable weights builds
on the previous concepts: we want to compute context vectors as weighted sums over
the input vectors specific to a certain input element. As you will see, there are only slight
differences compared to the basic self-attention mechanism we coded earlier.
The most notable difference is the introduction of weight matrices that are
updated during model training. These trainable weight matrices are crucial so that
the model (specifically, the attention module inside the model) can learn to produce
“good” context vectors.


## Computing the attention weights step by step
We will implement the self-attention mechanism step by step by introducing the
three trainable weight matrices Wq, Wk, and Wv. These three matrices are used to
project the embedded input tokens, x(i), into query, key, and value vectors, respec-
tively, as illustrated in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image44.png?raw=true)

In the first step of the self-attention mechanism with trainable weight matrices, we compute query (q), key (k), and value (v) vectors for input elements x. Similar to previous sections, we designate the second input, x(2), as the query input. The query vector q(2) is obtained via matrix multiplication between the input x(2) and the weight matrix Wq. Similarly, we obtain the key and value vectors via matrix multiplication involving the weight matrices Wk and Wv.
Earlier, we defined the second input element x(2) as the query when we computed the
simplified attention weights to compute the context vector z(2). Then we generalized
this to compute all context vectors z(1) ... z(T) for the six-word input sentence “Your
journey starts with one step.”
Similarly, we start here by computing only one context vector, z(2), for illustration
purposes. We will then modify this code to calculate all context vectors.
Let’s begin by defining a few variables:

```python
x_2 = inputs[1] #The second input element
d_in = inputs.shape[1] # The input embedding size, d=3
d_out = 2 # The output embedding size, d_out=2
```
Note that in GPT-like models, the input and output dimensions are usually the same,
but to better follow the computation, we’ll use different input ( d_in=3) and output
(d_out=2) dimensions here.
Next, we initialize the three weight matrices Wq, Wk, and Wv shown in figure above:
```python
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
```
We set requires_grad=False to reduce clutter in the outputs, but if we were to use
the weight matrices for model training, we would set requires_grad=True to update
these matrices during model training.
Next, we compute the query, key, and value vectors:
```python
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)
```
The output for the query results in a two-dimensional vector since we set the number
of columns of the corresponding weight matrix, via d_out, to 2:
tensor([0.4306, 1.4551])

**Weight parameters vs. attention weights**
In the weight matrices W, the term “weight” is short for “weight parameters,” the val-
ues of a neural network that are optimized during training. This is not to be confused
with the attention weights. As we already saw, attention weights determine the extent
to which a context vector depends on the different parts of the input (i.e., to what
extent the network focuses on different parts of the input).
In summary, weight parameters are the fundamental, learned coefficients that define
the network’s connections, while attention weights are dynamic, context-specific values.

Even though our temporary goal is only to compute the one context vector, z(2), we still
require the key and value vectors for all input elements as they are involved in com-
puting the attention weights with respect to the query q (2).
We can obtain all keys and values via matrix multiplication:
```python
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
```
As we can tell from the outputs, we successfully projected the six input tokens from a
three-dimensional onto a two-dimensional embedding space:

keys.shape: torch.Size([6, 2])
values.shape: torch.Size([6, 2])

The second step is to compute the attention scores, as shown in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image45.png?raw=true)

The attention score computation is a dot-product computation similar to what we used in the
simplified self-attention mechanism. The new aspect here is that we are not directly computing the dot-product between the input elements but using the query and key obtained by transforming the inputs via the respective weight matrices.

First, let’s compute the attention score ω22:

```python
keys_2 = keys[1] #Remember that Python starts indexing at 0.
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)
```
The result for the unnormalized attention score is
tensor(1.8524)
Again, we can generalize this computation to all attention scores via matrix
multiplication
```python
attn_scores_2 = query_2 @ keys.T #All attention scores for given query
print(attn_scores_2)
```

Now, we want to go from the attention scores to the attention weights, as illustrated in
figure below. We compute the attention weights by scaling the attention scores and
using the softmax function. However, now we scale the attention scores by dividing
them by the square root of the embedding dimension of the keys (taking the square
root is mathematically the same as exponentiating by 0.5):

```python
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
```
![alt text](https://github.com/Rezashatery/LLM/blob/main/image46.png?raw=true)

The resulting attention weights are
tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])

**The rationale behind scaled-dot product attention**
The reason for the normalization by the embedding dimension size is to improve the
training performance by avoiding small gradients. For instance, when scaling up the
embedding dimension, which is typically greater than 1,000 for GPT-like LLMs, large
dot products can result in very small gradients during backpropagation due to the
softmax function applied to them. As dot products increase, the softmax function
behaves more like a step function, resulting in gradients nearing zero. These small
gradients can drastically slow down learning or cause training to stagnate.
The scaling by the square root of the embedding dimension is the reason why this
self-attention mechanism is also called scaled-dot product attention.
Now, the final step is to compute the context vectors, as illustrated in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image47.png?raw=true)

Similar to when we computed the context vector as a weighted sum over the input vec-
tors, we now compute the context vector as a weighted sum over the
value vectors. Here, the attention weights serve as a weighting factor that weighs
the respective importance of each value vector. Also as before, we can use matrix mul-
tiplication to obtain the output in one step:
```python
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
```
The contents of the resulting vector are as follows:
tensor([0.3061, 0.8210])

So far, we’ve only computed a single context vector, z(2). Next, we will generalize the
code to compute all context vectors in the input sequence, z(1) to z(T).

**Why query, key, and value?**
The terms “key,” “query,” and “value” in the context of attention mechanisms are
borrowed from the domain of information retrieval and databases, where similar con-
cepts are used to store, search, and retrieve information.
A query is analogous to a search query in a database. It represents the current item
(e.g., a word or token in a sentence) the model focuses on or tries to understand.
The query is used to probe the other parts of the input sequence to determine how
much attention to pay to them.
The key is like a database key used for indexing and searching. In the attention mech-
anism, each item in the input sequence (e.g., each word in a sentence) has an asso-
ciated key. These keys are used to match the query.
The value in this context is similar to the value in a key-value pair in a database. It
represents the actual content or representation of the input items. Once the model
determines which keys (and thus which parts of the input) are most relevant to the
query (the current focus item), it retrieves the corresponding values.
The use of these matrices (𝑊𝑞,𝑊𝑘,𝑊𝑣) in attention mechanisms is designed to transform the input into better representations that are more suited for capturing relationships and dependencies within the data, which is critical for a wide range of tasks. 
Think of (𝑊𝑞,𝑊𝑘,𝑊𝑣) as analogous to lenses that allow the model to "view" the input in different ways:
𝑊𝑞 : A lens that defines what you're looking for (queries).
𝑊𝑘 : A lens that defines how to describe the data (keys).
𝑊𝑣 : A lens that determines what information to use (values).
These lenses are learned in a task-specific manner, enabling the model to adapt its focus based on what matters most for the task at hand.
By applying (𝑊𝑞,𝑊𝑘,𝑊𝑣):
The input is transformed into more useful representations for computing attention scores and aggregating information.
The model becomes more versatile and effective for a wide range of tasks.
It ensures that the attention mechanism can capture both local and global relationships, depending on the context.This flexibility and expressiveness are key reasons why attention mechanisms (and transformers) have become foundational in modern machine learning.



## Implementing a compact self-attention Python class
At this point, we have gone through a lot of steps to compute the self-attention out-
puts. We did so mainly for illustration purposes so we could go through one step at a
time. In practice, with the LLM implementation in the next chapter in mind, it is
helpful to organize this code into a Python class, as shown in the following listing.

```python
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key= nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec    
```
In this PyTorch code, SelfAttention_v1 is a class derived from nn.Module, which is a
fundamental building block of PyTorch models that provides necessary functionalities
for model layer creation and management.
The __init__ method initializes trainable weight matrices (W_query, W_key, and
W_value) for queries, keys, and values, each transforming the input dimension d_in to
an output dimension d_out.
During the forward pass, using the forward method, we compute the attention
scores (attn_scores) by multiplying queries and keys, normalizing these scores using
softmax. Finally, we create a context vector by weighting the values with these normal-
ized attention scores.
We can use this class as follows:
```python
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
```
Since inputs contains six embedding vectors, this results in a matrix storing the six
context vectors:
```python
tensor([[0.2996, 0.8053],
[0.3061, 0.8210],
[0.3058, 0.8203],
[0.2948, 0.7939],
[0.2927, 0.7891],
[0.2990, 0.8040]], grad_fn=<MmBackward0>)
```
As a quick check, notice that the second row ([0.3061, 0.8210]) matches the con-
tents of context_vec_2 in the previous section. Figure below summarizes the self-atten-
tion mechanism we just implemented.
Self-attention involves the trainable weight matrices Wq, Wk, and Wv. These matrices
transform input data into queries, keys, and values, respectively, which are crucial com-
ponents of the attention mechanism. As the model is exposed to more data during
training, it adjusts these trainable weights, as we will see in upcoming chapters.
We can improve the SelfAttention_v1 implementation further by utilizing
PyTorch’s nn.Linear layers, which effectively perform matrix multiplication when
the bias units are disabled. Additionally, a significant advantage of using nn.Linear instead of manually implementing nn.Parameter(torch.rand(...)) is that nn.Linear
has an optimized weight initialization scheme, contributing to more stable and
effective model training.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image48.png?raw=true)

In self-attention, we transform the input vectors in the input matrix X with the three weight
matrices, Wq, Wk, and Wv. The new compute the attention weight matrix based on the resulting queries (Q) and
keys (K). Using the attention weights and values (V), we then compute the context vectors (Z). For visual clarity,
we focus on a single input text with n tokens, not a batch of multiple inputs. Consequently, the three-dimensional
input tensor is simplified to a two-dimensional matrix in this context. This approach allows for a more straightforward
visualization and understanding of the processes involved. For consistency with later figures, the values in the
attention matrix do not depict the real attention weights. (The numbers in this figure are truncated to two digits
after the decimal point to reduce visual clutter. The values in each row should add up to 1.0 or 100%.)
```python
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec   
```

You can use the SelfAttention_v2 similar to SelfAttention_v1:
```python
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
```
The output is
```python
tensor([[-0.0739,
[-0.0748,
[-0.0749,
[-0.0760,
[-0.0763,
[-0.0754,
0.0713],
0.0703],
0.0702],
0.0685],
0.0679],
0.0693]], grad_fn=<MmBackward0>)
```
Note that SelfAttention_v1 and SelfAttention_v2 give different outputs because
they use different initial weights for the weight matrices since nn.Linear uses a more
sophisticated weight initialization scheme.
Next, we will make enhancements to the self-attention mechanism, focusing specifically
on incorporating causal and multi-head elements. The causal aspect involves modify-
ing the attention mechanism to prevent the model from accessing future information
in the sequence, which is crucial for tasks like language modeling, where each word
prediction should only depend on previous words.
The multi-head component involves splitting the attention mechanism into multi-
ple “heads.” Each head learns different aspects of the data, allowing the model to
simultaneously attend to information from different representation subspaces at dif-
ferent positions. This improves the model’s performance in complex tasks.


## Hiding future words with causal attention
For many LLM tasks, you will want the self-attention mechanism to consider only the
tokens that appear prior to the current position when predicting the next token in a
sequence. **Causal attention**, also known as masked attention, is a specialized form of self-
attention. It restricts a model to only consider previous and current inputs in a sequence
when processing any given token when computing attention scores. This is in contrast
to the standard self-attention mechanism, which allows access to the entire input
sequence at once.
Now, we will modify the standard self-attention mechanism to create a causal
attention mechanism, which is essential for developing an LLM in the subsequent
chapters. To achieve this in GPT-like LLMs, for each token processed, we mask out
the future tokens, which come after the current token in the input text, as illus-
trated in figure below. We mask out the attention weights above the diagonal, and we normalize the nonmasked attention weights such that the attention weights sum to 1 in
each row.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image49.png?raw=true)

In causal attention, we mask out the attention weights above the diagonal such that for
a given input, the LLM can’t access future tokens when computing the context vectors using the
attention weights. For example, for the word “journey” in the second row, we only keep the attention
weights for the words before (“Your”) and in the current position (“journey”).

## Applying a causal attention mask

Our next step is to implement the causal attention mask in code. To implement the
steps to apply a causal attention mask to obtain the masked attention weights, as sum-
marized in figure below, let’s work with the attention scores and weights from the previ-
ous section to code the causal attention mechanism.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image50.png?raw=true)

One way to obtain the masked attention weight matrix in causal attention is to apply the
softmax function to the attention scores, zeroing out the elements above the diagonal and normalizing the resulting matrix.
In the first step, we compute the attention weights using the softmax function as we
have done previously:

```python
queries = sa_v2.W_query(inputs) #Reuses the query and key weight matrices of the SelfAttention_v2 object from the previous section for convenience
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)
```

We can implement the second step using PyTorch’s tril function to create a mask
where the values above the diagonal are zero:
```python
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)
```
The resulting mask is
```python
tensor([[1., 0., 0., 0., 0., 0.],
[1., 1., 0., 0., 0., 0.],
[1., 1., 1., 0., 0., 0.],
[1., 1., 1., 1., 0., 0.],
[1., 1., 1., 1., 1., 0.],
[1., 1., 1., 1., 1., 1.]])
```
Now, we can multiply this mask with the attention weights to zero-out the values above
the diagonal:
```python
masked_simple = attn_weights*mask_simple
print(masked_simple)
```
As we can see, the elements above the diagonal are successfully zeroed out:
```python
tensor([[0.1921, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
[0.2041, 0.1659, 0.0000, 0.0000, 0.0000, 0.0000],
[0.2036, 0.1659, 0.1662, 0.0000, 0.0000, 0.0000],
[0.1869, 0.1667, 0.1668, 0.1571, 0.0000, 0.0000],
[0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.0000],
[0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
grad_fn=<MulBackward0>)
```
The third step is to renormalize the attention weights to sum up to 1 again in each
row. We can achieve this by dividing each element in each row by the sum in each row:
```python
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
```
The result is an attention weight matrix where the attention weights above the diago-
nal are zeroed-out, and the rows sum to 1:
```python
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
[0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
[0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
[0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
[0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
[0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
grad_fn=<DivBackward0>)
```

**Information leakage**
When we apply a mask and then renormalize the attention weights, it might initially
appear that information from future tokens (which we intend to mask) could still influ-
ence the current token because their values are part of the softmax calculation. How-
ever, the key insight is that when we renormalize the attention weights after masking,
what we’re essentially doing is recalculating the softmax over a smaller subset (since
masked positions don’t contribute to the softmax value).
The mathematical elegance of softmax is that despite initially including all positions
in the denominator, after masking and renormalizing, the effect of the masked posi-
tions is nullified—they don’t contribute to the softmax score in any meaningful way.
In simpler terms, after masking and renormalization, the distribution of attention
weights is as if it was calculated only among the unmasked positions to begin with.
This ensures there’s no information leakage from future (or otherwise masked)
tokens as we intended.

While we could wrap up our implementation of causal attention at this point, we can
still improve it. Let’s take a mathematical property of the softmax function and imple-
ment the computation of the masked attention weights more efficiently in fewer steps,
as shown in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image51.png?raw=true)

A more efficient way to obtain the masked attention weight matrix in
causal attention is to mask the attention scores with negative infinity values before
applying the softmax function.

The softmax function converts its inputs into a probability distribution. When nega-
tive infinity values (-∞) are present in a row, the softmax function treats them as zero
probability. (Mathematically, this is because e –∞ approaches 0.)
We can implement this more efficient masking “trick” by creating a mask with 1s
above the diagonal and then replacing these 1s with negative infinity (-inf) values:
```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
```
This results in the following mask:
```python
tensor([[0.2899,-inf,-inf,-inf,-inf,-inf],
[0.4656, 0.1723,-inf,-inf,-inf,-inf],
[0.4594, 0.1703, 0.1731,-inf,-inf,-inf],
[0.2642, 0.1024, 0.1036, 0.0186,-inf,-inf],
[0.2183, 0.0874, 0.0882, 0.0177, 0.0786,-inf],
[0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]],
grad_fn=<MaskedFillBackward0>)
```
Now all we need to do is apply the softmax function to these masked results, and we
are done:   
```python
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)
```
As we can see based on the output, the values in each row sum to 1, and no further
normalization is necessary:
```python
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
[0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
[0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
[0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
[0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
[0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
grad_fn=<SoftmaxBackward0>)
```
We could now use the modified attention weights to compute the context vectors via
context_vec = attn_weights @ values, as in previous section. However, we will first cover
another minor tweak to the causal attention mechanism that is useful for reducing
overfitting when training LLMs.

## Masking additional attention weights with dropout
Dropout in deep learning is a technique where randomly selected hidden layer units
are ignored during training, effectively “dropping” them out. This method helps pre-
vent overfitting by ensuring that a model does not become overly reliant on any spe-
cific set of hidden layer units. It’s important to emphasize that dropout is only used
during training and is disabled afterward.
In the transformer architecture, including models like GPT, dropout in the atten-
tion mechanism is typically applied at two specific times: after calculating the atten-
tion weights or after applying the attention weights to the value vectors. Here we will
apply the dropout mask after computing the attention weights, as illustrated in fig-
ure below, because it’s the more common variant in practice.
In the following code example, we use a dropout rate of 50%, which means mask-
ing out half of the attention weights. (When we train the GPT model in later chapters,
we will use a lower dropout rate, such as 0.1 or 0.2.) We apply PyTorch’s dropout
implementation first to a 6 × 6 tensor consisting of 1s for simplicity:

```python
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6) # Here, we create a matrix of 1s.
print(dropout(example))
```
![alt text](https://github.com/Rezashatery/LLM/blob/main/image52.png?raw=true)

Using the causal attention mask (upper left), we apply an additional
dropout mask (upper right) to zero out additional attention weights to reduce overfitting
during training.
As we can see, approximately half of the values are zeroed out:
```python
tensor([[2., 2., 0., 2., 2., 0.],
[0., 0., 0., 2., 0., 2.],
[2., 2., 2., 2., 0., 2.],
[0., 2., 2., 0., 0., 2.],
[0., 2., 0., 2., 0., 2.],
[0., 2., 2., 2., 2., 0.]])
```
When applying dropout to an attention weight matrix with a rate of 50%, half of the
elements in the matrix are randomly set to zero. To compensate for the reduction in
active elements, the values of the remaining elements in the matrix are scaled up by a
factor of 1/0.5 = 2. This scaling is crucial to maintain the overall balance of the attention weights, ensuring that the average influence of the attention mechanism remains
consistent during both the training and inference phases.
Now let’s apply dropout to the attention weight matrix itself:
```python
torch.manual_seed(123)
print(dropout(attn_weights))
```
The resulting attention weight matrix now has additional elements zeroed out and the
remaining 1s rescaled:

```python
tensor([[2.0000, 0.0000, 0 .0000, 0.0000, 0.0000, 0.0000],
[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
[0.7599, 0.6194, 0.6206, 0.0000, 0.0000, 0.0000],
[0.0000, 0.4921, 0.4925, 0.0000, 0.0000, 0.0000],
[0.0000, 0.3966, 0.0000, 0.3775, 0.0000, 0.0000],
[0.0000, 0.3327, 0.3331, 0.3084, 0.3331, 0.0000]],
grad_fn=<MulBackward0>)
```
Having gained an understanding of causal attention and dropout masking, we can
now develop a concise Python class. This class is designed to facilitate the efficient
application of these two techniques.


## Implementing a compact causal attention class
We will now incorporate the causal attention and dropout modifications into the
SelfAttention Python class we developed in previous section. This class will then serve as a
template for developing multi-head attention, which is the final attention class we will
implement.
But before we begin, let’s ensure that the code can handle batches consisting of
more than one input so that the CausalAttention class supports the batch outputs
produced by the data loader we implemented in previous chapter. 
For simplicity, to simulate such batch inputs, we duplicate the input text example:
```python
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape) # wo inputs with six tokens each; each token has embedding dimension 3.
```
This results in a three-dimensional tensor consisting of two input texts with six tokens
each, where each token is a three-dimensional embedding vector:
torch.Size([2, 6, 3])
The following CausalAttention class is similar to the SelfAttention class we imple-
mented earlier, except that we added the dropout and causal mask components.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image53.png?raw=true)

While all added code lines should be familiar at this point, we now added a self
.register_buffer() call in the __init__ method. The use of register_buffer in
PyTorch is not strictly necessary for all use cases but offers several advantages here. For
instance, when we use the CausalAttention class in our LLM, buffers are automati-
cally moved to the appropriate device (CPU or GPU) along with our model, which will
be relevant when training our LLM. This means we don’t need to manually ensure
these tensors are on the same device as your model parameters, avoiding device mis-
match errors.
We can use the CausalAttention class as follows, similar to SelfAttention
previously:
```python
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
```
The resulting context vector is a three-dimensional tensor where each token is now
represented by a two-dimensional embedding:
context_vecs.shape: torch.Size([2, 6, 2])

![alt text](https://github.com/Rezashatery/LLM/blob/main/image54.png?raw=true)

Here’s what we’ve done so far. We began with a simplified attention mechanism, added trainable
weights, and then added a causal attention mask. Next, we will extend the causal attention mechanism and code multi-head attention, which we will use in our LLM.

## Extending single-head attention to multi-head attention

Our final step will be to extend the previously implemented causal attention class over
multiple heads. This is also called multi-head attention.
The term “multi-head” refers to dividing the attention mechanism into multiple
“heads,” each operating independently. In this context, a single causal attention mod-
ule can be considered single-head attention, where there is only one set of attention
weights processing the input sequentially.
We will tackle this expansion from causal attention to multi-head attention. First,
we will intuitively build a multi-head attention module by stacking multiple Causal-
Attention modules. Then we will then implement the same multi-head attention
module in a more complicated but more computationally efficient way.


## Stacking multiple single-head attention layers
In practical terms, implementing multi-head attention involves creating multiple
instances of the self-attention mechanism, each with its own weights,
and then combining their outputs. Using multiple instances of the self-attention
mechanism can be computationally intensive, but it’s crucial for the kind of complex
pattern recognition that models like transformer-based LLMs are known for.

Figure 3.24 illustrates the structure of a multi-head attention module, which con-
sists of multiple single-head attention modules, as previously depicted in previous figure,
stacked on top of each other.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image55.png?raw=true)

The multi-head attention module includes two single-head attention modules stacked on top of
each other. So, instead of using a single matrix Wv for computing the value matrices, in a multi-head attention module with two heads, we now have two value weight matrices: Wv1 and Wv2. The same applies to the other weight matrices, WQ and Wk. We obtain two sets of context vectors Z1 and Z2 that we can combine into a single context vector matrix Z.

As mentioned before, the main idea behind multi-head attention is to run the attention
mechanism multiple times (in parallel) with different, learned linear projections—the
results of multiplying the input data (like the query, key, and value vectors in attention
mechanisms) by a weight matrix. In code, we can achieve this by implementing a sim-
ple MultiHeadAttentionWrapper class that stacks multiple instances of our previously
implemented CausalAttention module.

```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
        [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
        for _ in range(num_heads)])
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
```
For example, if we use this MultiHeadAttentionWrapper class with two attention heads
(via num_heads=2) and CausalAttention output dimension d_out=2, we get a four-
dimensional context vector (d_out*num_heads=4), as depicted in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image56.png?raw=true)

Using the MultiHeadAttentionWrapper, we specified the number of
attention heads (num_heads). If we set num_heads=2, as in this example, we obtain
a tensor with two sets of context vector matrices. In each context vector matrix, the
rows represent the context vectors corresponding to the tokens, and the columns
correspond to the embedding dimension specified via d_out=4. We concatenate these
context vector matrices along the column dimension. Since we have two attention
heads and an embedding dimension of 2, the final embedding dimension is 2 × 2 = 4.

To illustrate this further with a concrete example, we can use the MultiHeadAttention-
Wrapper class similar to the CausalAttention class before:
```python
torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```
The first dimension of the resulting context_vecs tensor is 2 since we have two input
texts (the input texts are duplicated, which is why the context vectors are exactly the
same for those). The second dimension refers to the 6 tokens in each input. The third
dimension refers to the four-dimensional embedding of each token.
Up to this point, we have implemented a MultiHeadAttentionWrapper that combined
multiple single-head attention modules. However, these are processed sequentially via
[head(x) for head in self.heads] in the forward method. We can improve this
implementation by processing the heads in parallel. One way to achieve this is by com-
puting the outputs for all attention heads simultaneously via matrix multiplication.

## Implementing multi-head attention with weight splits
So far, we have created a MultiHeadAttentionWrapper to implement multi-head
attention by stacking multiple single-head attention modules. This was done by instan-
tiating and combining several CausalAttention objects.
Instead of maintaining two separate classes, MultiHeadAttentionWrapper and
CausalAttention, we can combine these concepts into a single MultiHeadAttention
class. Also, in addition to merging the MultiHeadAttentionWrapper with the Causal-
Attention code, we will make some other modifications to implement multi-head
attention more efficiently.
In the MultiHeadAttentionWrapper, multiple heads are implemented by creating
a list of CausalAttention objects (self.heads), each representing a separate atten-
tion head. The CausalAttention class independently performs the attention mecha-
nism, and the results from each head are concatenated. In contrast, the following
MultiHeadAttention class integrates the multi-head functionality within a single class.
It splits the input into multiple heads by reshaping the projected query, key, and value
tensors and then combines the results from these heads after computing attention.
Let’s take a look at the MultiHeadAttention class before we discuss it further.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image57.png?raw=true)
![alt text](https://github.com/Rezashatery/LLM/blob/main/image58.png?raw=true)

Even though the reshaping (.view) and transposing (.transpose) of tensors inside
the MultiHeadAttention class looks very mathematically complicated, the Multi-
HeadAttention class implements the same concept as the MultiHeadAttention-
Wrapper earlier.
On a big-picture level, in the previous MultiHeadAttentionWrapper, we stacked
multiple single-head attention layers that we combined into a multi-head attention
layer. The MultiHeadAttention class takes an integrated approach. It starts with a
multi-head layer and then internally splits this layer into individual attention heads, as
illustrated in figure below.
The splitting of the query, key, and value tensors is achieved through tensor reshap-
ing and transposing operations using PyTorch’s .view and .transpose methods. The
input is first transformed (via linear layers for queries, keys, and values) and then
reshaped to represent multiple heads.
The key operation is to split the d_out dimension into num_heads and head_dim,
where head_dim = d_out / num_heads. This splitting is then achieved using the .view
method: a tensor of dimensions (b, num_tokens, d_out) is reshaped to dimension
(b, num_tokens, num_heads, head_dim).

![alt text](https://github.com/Rezashatery/LLM/blob/main/image59.png?raw=true)

In the MultiHeadAttentionWrapper class with two attention heads,
we initialized two weight matrices, Wq1 and Wq2, and computed two query matrices, Q1
and Q2 (top). In the MultiheadAttention class, we initialize one larger weight matrix
Wq, only perform one matrix multiplication with the inputs to obtain a query matrix Q, and
then split the query matrix into Q1 and Q2 (bottom). We do the same for the keys and
values, which are not shown to reduce visual clutter.
The tensors are then transposed to bring the num_heads dimension before the num_
tokens dimension, resulting in a shape of (b, num_heads, num_tokens, head_dim). This
transposition is crucial for correctly aligning the queries, keys, and values across the
different heads and performing batched matrix multiplications efficiently.
To illustrate this batched matrix multiplication, suppose we have the following
tensor:

```python
# The shape of this tensor is (b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4).
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573], 
[0.8993, 0.0390, 0.9268, 0.7388],
[0.7179, 0.7058, 0.9156, 0.4340]],
[[0.0772, 0.3565, 0.1479, 0.5331],
[0.4066, 0.2318, 0.4545, 0.9737],
[0.4606, 0.5159, 0.4220, 0.5786]]]])
```
Now we perform a batched matrix multiplication between the tensor itself and a view
of the tensor where we transposed the last two dimensions, num_tokens and head_dim:

print(a @ a.transpose(2, 3))
The result is
```python
tensor([[[[1.3208, 1.1631, 1.2879],
[1.1631, 2.2150, 1.8424],
[1.2879, 1.8424, 2.0402]],
[[0.4391, 0.7003, 0.5903],
[0.7003, 1.3737, 1.0620],
[0.5903, 1.0620, 0.9912]]]])
```
In this case, the matrix multiplication implementation in PyTorch handles the four-
dimensional input tensor so that the matrix multiplication is carried out between the two
last dimensions (num_tokens, head_dim) and then repeated for the individual heads.
For instance, the preceding becomes a more compact way to compute the matrix
multiplication for each head separately:
```python
first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
print("First head:\n", first_res)
second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
print("\nSecond head:\n", second_res)
```
The results are exactly the same results as those we obtained when using the batched
matrix multiplication print(a @ a.transpose(2, 3))

Continuing with MultiHeadAttention, after computing the attention weights and con-
text vectors, the context vectors from all heads are transposed back to the shape (b,
num_tokens, num_heads, head_dim). These vectors are then reshaped (flattened) into
the shape (b, num_tokens, d_out), effectively combining the outputs from all heads.
Additionally, we added an output projection layer (self.out_proj) to Multi-
HeadAttention after combining the heads, which is not present in the Causal-
Attention class. This output projection layer is not strictly necessary,but it is commonly used in many LLM architectures, which is why I
added it here for completeness.

Even though the MultiHeadAttention class looks more complicated than the
MultiHeadAttentionWrapper due to the additional reshaping and transposition of
tensors, it is more efficient. The reason is that we only need one matrix multiplication
to compute the keys, for instance, keys = self.W_key(x) (the same is true for the que-
ries and values). In the MultiHeadAttentionWrapper, we needed to repeat this matrix
multiplication, which is computationally one of the most expensive steps, for each
attention head. 
The MultiHeadAttention class can be used similar to the SelfAttention and
CausalAttention classes we implemented earlier:
```python
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```
For comparison, the smallest GPT-2 model (117 million parameters) has 12 atten-
tion heads and a context vector embedding size of 768. The largest GPT-2 model (1.5
billion parameters) has 25 attention heads and a context vector embedding size of
1,600. The embedding sizes of the token inputs and context embeddings are the same
in GPT models (d_in = d_out).

# Implementing a GPT model from scratch to generate text (CHAPTER 4)

You’ve already learned and coded the multi-head attention mechanism, one of the
core components of LLMs. Now, we will code the other building blocks of an LLM
and assemble them into a GPT-like model that we will train in the next chapter to
generate human-like text.
The LLM architecture referenced in figure below, consists of several building
blocks. We will begin with a top-down view of the model architecture before cover-
ing the individual components in more detail.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image60.png?raw=true)



## Coding an LLM architecture
LLMs, such as GPT (which stands for generative pretrained transformer), are large deep
neural network architectures designed to generate new text one word (or token) at a
time. However, despite their size, the model architecture is less complicated than you
might think, since many of its components are repeated, as we will see later. Figure below.
provides a top-down view of a GPT-like LLM, with its main components highlighted.
We have already covered several aspects of the LLM architecture, such as input
tokenization and embedding and the masked multi-head attention module. Now, we
will implement the core structure of the GPT model, including its transformer blocks,
which we will later train to generate human-like text.
Previously, we used smaller embedding dimensions for simplicity, ensuring that the
concepts and examples could comfortably fit on a single page. Now, we are scaling up
to the size of a small GPT-2 model, specifically the smallest version with 124 million
parameters, as described in “Language Models Are Unsupervised Multitask Learners,”
by Radford et al. (https://mng.bz/yoBq). Note that while the original report men-
tions 117 million parameters, this was later corrected. In chapter 6, we will focus on
loading pretrained weights into our implementation and adapting it for larger GPT-2
models with 345, 762, and 1,542 million parameters.
In the context of deep learning and LLMs like GPT, the term “parameters” refers
to the trainable weights of the model. These weights are essentially the internal vari-
ables of the model that are adjusted and optimized during the training process to
minimize a specific loss function. This optimization allows the model to learn from
the training data.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image61.png?raw=true)

We specify the configuration of the small GPT-2 model via the following Python dictio-
nary, which we will use in the code examples later:

```python
GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024,
"emb_dim": 768,  # Embedding dimension
"n_heads": 12,   # Number of attention heads
"n_layers": 12,  # Number of layers
"drop_rate": 0.1, # Number of layers
"qkv_bias": False # Query-Key-Value bias
}
```

In the GPT_CONFIG_124M dictionary, we use concise variable names for clarity and to
prevent long lines of code:
- vocab_size refers to a vocabulary of 50,257 words, as used by the BPE tokenizer
(see chapter 2).
- context_length denotes the maximum number of input tokens the model can
handle via the positional embeddings (see chapter 2).
- emb_dim represents the embedding size, transforming each token into a 768-
dimensional vector.
- n_heads indicates the count of attention heads in the multi-head attention
mechanism (see chapter 3).
- n_layers specifies the number of transformer blocks in the model, which we
will cover in the upcoming discussion.
- drop_rate indicates the intensity of the dropout mechanism (0.1 implies a 10%
random drop out of hidden units) to prevent overfitting (see chapter 3).
- qkv_bias determines whether to include a bias vector in the Linear layers of
the multi-head attention for query, key, and value computations. We will initially
disable this, following the norms of modern LLMs, but we will revisit it in chap-
ter 6 when we load pretrained GPT-2 weights from OpenAI into our model (see
chapter 6).

Using this configuration, we will implement a GPT placeholder architecture (Dummy-
GPTModel), as shown in figure below. This will provide us with a big-picture view of how
everything fits together and what other components we need to code to assemble the
full GPT model architecture.
The numbered boxes in figure below illustrate the order in which we tackle the indi-
vidual concepts required to code the final GPT architecture. We will start with step 1,
a placeholder GPT backbone we will call DummyGPTModel.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image62.png?raw=true)

```python
import torch
import torch.nn as nn
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg)
                for _ in range(cfg["n_layers"])])
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
        torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    def forward(self, x):
        return x
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    def forward(self, x):
        return x    
```
The DummyGPTModel class in this code defines a simplified version of a GPT-like
model using PyTorch’s neural network module (nn.Module). The model architecture
in the DummyGPTModel class consists of token and positional embeddings, dropout,
a series of transformer blocks (DummyTransformerBlock), a final layer normalization
(DummyLayerNorm), and a linear output layer (out_head). The configuration is
passed in via a Python dictionary, for instance, the GPT_CONFIG_124M dictionary we
created earlier.
The forward method describes the data flow through the model: it computes token
and positional embeddings for the input indices, applies dropout, processes the data
through the transformer blocks, applies normalization, and finally produces logits
with the linear output layer.
The code in above is already functional. However, for now, note that we use
placeholders (DummyLayerNorm and DummyTransformerBlock) for the transformer block
and layer normalization, which we will develop later.
Next, we will prepare the input data and initialize a new GPT model to illustrate
its usage. Building on our coding of the tokenizer (see chapter 2), let’s now con-
sider a high-level overview of how data flows in and out of a GPT model, as shown in
figure below.
To implement these steps, we tokenize a batch consisting of two text inputs for the
GPT model using the tiktoken tokenizer from chapter 2:

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)
```
![alt text](https://github.com/Rezashatery/LLM/blob/main/image63.png?raw=true)

A big-picture overview showing how the input data is tokenized, embedded, and fed to the GPT model. Note that in our DummyGPTClass coded earlier, the token embedding is handled inside the GPT model. In LLMs, the embedded input token dimension typically matches the output dimension. The output embeddings here represent the context vectors.

The resulting token IDs for the two texts are as follows:
```python
tensor([[6109,3626,6100,345],
        [6109,1110,6622,257]])
```
The first row corresponds to the first text, and the second row corresponds to the second text.
Next, we initialize a new 124-million-parameter DummyGPTModel instance and feed it
the tokenized batch:
```python
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)
```
The model outputs, which are commonly referred to as logits, are as follows:
```python
Output shape: torch.Size([2, 4, 50257])
tensor([[[-1.2034, 0.3201, -0.7130, ..., -1.5548, -0.2390, -0.4667],
[-0.1192, 0.4539, -0.4432, ..., 0.2392, 1.3469, 1.2430],
[ 0.5307, 1.6720, -0.4695, ..., 1.1966, 0.0111, 0.5835],
[ 0.0139, 1.6755, -0.3388, ..., 1.1586, -0.0435, -1.0400]],
[[-1.0908, 0.1798, -0.9484, ..., -1.6047, 0.2439, -0.4530],
[-0.7860, 0.5581, -0.0610, ..., 0.4835, -0.0077, 1.6621],
[ 0.3567, 1.2698, -0.6398, ..., -0.0162, -0.1296, 0.3717],
[-0.2407, -0.7349, -0.5102, ..., 2.0057, -0.3694, 0.1814]]],
grad_fn=<UnsafeViewBackward0>)
```
The output tensor has two rows corresponding to the two text samples. Each text sam-
ple consists of four tokens; each token is a 50,257-dimensional vector, which matches
the size of the tokenizer’s vocabulary.
The embedding has 50,257 dimensions because each of these dimensions refers to
a unique token in the vocabulary. When we implement the postprocessing code, we
will convert these 50,257-dimensional vectors back into token IDs, which we can then
decode into words.
Now that we have taken a top-down look at the GPT architecture and its inputs and
outputs, we will code the individual placeholders, starting with the real layer normal-
ization class that will replace the DummyLayerNorm in the previous code.




## Normalizing activations with layer normalization
Training deep neural networks with many layers can sometimes prove challenging
due to problems like vanishing or exploding gradients. These problems lead to unsta-
ble training dynamics and make it difficult for the network to effectively adjust its
weights, which means the learning process struggles to find a set of parameters
(weights) for the neural network that minimizes the loss function. In other words, the
network has difficulty learning the underlying patterns in the data to a degree that
would allow it to make accurate predictions or decisions.
Let’s now implement layer normalization to improve the stability and efficiency of neu-
ral network training. The main idea behind layer normalization is to adjust the activa-
tions (outputs) of a neural network layer to have a mean of 0 and a variance of 1, also
known as unit variance. This adjustment speeds up the convergence to effective
weights and ensures consistent, reliable training. In GPT-2 and modern transformer
architectures, layer normalization is typically applied before and after the multi-head
attention module, and, as we have seen with the DummyLayerNorm placeholder, before the final output layer. Figure below provides a visual overview of how layer normalization
functions.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image64.png?raw=true)

We can recreate the example shown in figure above via the following code, where we
implement a neural network layer with five inputs and six outputs that we apply to two
input examples:

```python
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)
```
This prints the following tensor, where the first row lists the layer outputs for the first
input and the second row lists the layer outputs for the second row:
```python
tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
[0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
grad_fn=<ReluBackward0>)
```
Before we apply layer normalization to these outputs, let’s examine the mean and
variance:

```python
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)
```
The output is
```python
Mean:
tensor([[0.1324],
[0.2170]], grad_fn=<MeanBackward1>)
Variance:
tensor([[0.0231],
[0.0398]], grad_fn=<VarBackward0>)
```

The first row in the mean tensor here contains the mean value for the first input row,
and the second output row contains the mean for the second input row.
Using keepdim=True in operations like mean or variance calculation ensures that the
output tensor retains the same number of dimensions as the input tensor, even though
the operation reduces the tensor along the dimension specified via dim. For instance,
without keepdim=True, the returned mean tensor would be a two-dimensional vector
[0.1324, 0.2170] instead of a 2 × 1–dimensional matrix [[0.1324], [0.2170]].
The dim parameter specifies the dimension along which the calculation of the statis-
tic (here, mean or variance) should be performed in a tensor. As figure below explains, for
a two-dimensional tensor (like a matrix), using dim=-1 for operations such as mean or
variance calculation is the same as using dim=1. This is because -1 refers to the tensor’s
last dimension, which corresponds to the columns in a two-dimensional tensor. Later,
when adding layer normalization to the GPT model, which produces three-dimensional
tensors with the shape [batch_size, num_tokens, embedding_size], we can still use
dim=-1 for normalization across the last dimension, avoiding a change from dim=1 to
dim=2.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image65.png?raw=true)

An illustration of the dim parameter when calculating the mean of a tensor. For instance, if we have a two-dimensional tensor (matrix) with dimensions [rows, columns], using dim=0 will perform the operation across rows (vertically, as shown at the bottom), resulting in an output that aggregates the data for each column. Using dim=1 or dim=-1 will perform the operation across columns (horizontally, as shown at the top), resulting in an output aggregating the data for each row.

Next, let’s apply layer normalization to the layer outputs we obtained earlier. The
operation consists of subtracting the mean and dividing by the square root of the vari-
ance (also known as the standard deviation):

```python
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)
```
As we can see based on the results, the normalized layer outputs, which now also con-
tain negative values, have 0 mean and a variance of 1:

```python
Normalized layer outputs:
tensor([[ 0.6159, 1.4126, -0.8719, 0.5872, -0.8719, -0.8719],
[-0.0189, 0.1121, -1.0876, 1.5173, 0.5647, -1.0876]],
grad_fn=<DivBackward0>)
Mean:
tensor([[-5.9605e-08],
[1.9868e-08]], grad_fn=<MeanBackward1>)
Variance:
tensor([[1.],
[1.]], grad_fn=<VarBackward0>)
```

Note that the value –5.9605e-08 in the output tensor is the scientific notation for
–5.9605 × 10-8, which is –0.000000059605 in decimal form. This value is very close to 0,
but it is not exactly 0 due to small numerical errors that can accumulate because of
the finite precision with which computers represent numbers.
To improve readability, we can also turn off the scientific notation when printing
tensor values by setting sci_mode to False:

```python
torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)
```
The output is
```python
Mean:
tensor([[
[
0.0000],
0.0000]], grad_fn=<MeanBackward1>)
Variance:
tensor([[1.],
[1.]], grad_fn=<VarBackward0>)
```

So far, we have coded and applied layer normalization in a step-by-step process. Let’s
now encapsulate this process in a PyTorch module that we can use in the GPT model
later.

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```
This specific implementation of layer normalization operates on the last dimension of
the input tensor x, which represents the embedding dimension (emb_dim). The vari-
able eps is a small constant (epsilon) added to the variance to prevent division by zero
during normalization. The scale and shift are two trainable parameters (of the
same dimension as the input) that the LLM automatically adjusts during training if it
is determined that doing so would improve the model’s performance on its training
task. This allows the model to learn appropriate scaling and shifting that best suit the
data it is processing.

**Biased variance**

In our variance calculation method, we use an implementation detail by setting
unbiased=False. For those curious about what this means, in the variance calcula-
tion, we divide by the number of inputs n in the variance formula. This approach does
not apply Bessel’s correction, which typically uses n – 1 instead of n in the denomi-
nator to adjust for bias in sample variance estimation. This decision results in a so-
called biased estimate of the variance. For LLMs, where the embedding dimension n
is significantly large, the difference between using n and n – 1 is practically negligible.
I chose this approach to ensure compatibility with the GPT-2 model’s normalization
layers and because it reflects TensorFlow’s default behavior, which was used to
implement the original GPT-2 model. Using a similar setting ensures our method is
compatible with the pretrained weights we will load in chapter 6.
Let’s now try the LayerNorm module in practice and apply it to the batch input:

```python
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)
```
The results show that the layer normalization code works as expected and normalizes
the values of each of the two inputs such that they have a mean of 0 and a variance of 1:

```python
Mean:
tensor([[
-0.0000],
[
0.0000]], grad_fn=<MeanBackward1>)
Variance:
tensor([[1.0000],
[1.0000]], grad_fn=<VarBackward0>)
```
We have now covered two of the building blocks we will need to implement the GPT
architecture, as shown in figure below. Next, we will look at the GELU activation func-
tion, which is one of the activation functions used in LLMs, instead of the traditional
ReLU function we used previously.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image66.png?raw=true)

**Layer normalization vs. batch normalization**

If you are familiar with batch normalization, a common and traditional normalization
method for neural networks, you may wonder how it compares to layer normalization.
Unlike batch normalization, which normalizes across the batch dimension, layer nor-
malization normalizes across the feature dimension. LLMs often require significant
computational resources, and the available hardware or the specific use case can
dictate the batch size during training or inference. Since layer normalization normal-
izes each input independently of the batch size, it offers more flexibility and stability
in these scenarios. This is particularly beneficial for distributed training or when
deploying models in environments where resources are constrained.


## Implementing a feed forward network with GELU activations
Next, we will implement a small neural network submodule used as part of the trans-
former block in LLMs. We begin by implementing the GELU activation function,
which plays a crucial role in this neural network submodule.
For additional information on implementing neural networks in
PyTorch, see section A.5 in appendix A.
NOTE
Historically, the ReLU activation function has been commonly used in deep learning
due to its simplicity and effectiveness across various neural network architectures.
However, in LLMs, several other activation functions are employed beyond the tradi-
tional ReLU. Two notable examples are GELU (Gaussian error linear unit) and SwiGLU
(Swish-gated linear unit).
GELU and SwiGLU are more complex and smooth activation functions incorpo-
rating Gaussian and sigmoid-gated linear units, respectively. They offer improved per-
formance for deep learning models, unlike the simpler ReLU.
The GELU activation function can be implemented in several ways; the exact ver-
sion is defined as GELU(x) = x⋅Φ(x), where Φ(x) is the cumulative distribution func-
tion of the standard Gaussian distribution. In practice, however, it’s common to
implement a computationally cheaper approximation (the original GPT-2 model was
also trained with this approximation, which was found via curve fitting):

![alt text](https://github.com/Rezashatery/LLM/blob/main/image67.png?raw=true)

In code, we can implement this function as a PyTorch module.

```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) *
        (x + 0.044715 * torch.pow(x, 3))
        ))
```
Next, to get an idea of what this GELU function looks like and how it compares to the
ReLU function, let’s plot these functions side by side:

```python
import matplotlib.pyplot as plt
gelu, relu = GELU(), nn.ReLU()
x = torch.linspace(-3, 3, 100) # Creates 100 sample data points in the range –3 to 3
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.show()
```
As we can see in the resulting plot in figure below, ReLU (right) is a piecewise linear
function that outputs the input directly if it is positive; otherwise, it outputs zero.
GELU (left) is a smooth, nonlinear function that approximates ReLU but with a non-
zero gradient for almost all negative values (except at approximately x = –0.75).

![alt text](https://github.com/Rezashatery/LLM/blob/main/image68.png?raw=true)
The smoothness of GELU can lead to better optimization properties during training,
as it allows for more nuanced adjustments to the model’s parameters. In contrast,
ReLU has a sharp corner at zero (figure 4.18, right), which can sometimes make opti-
mization harder, especially in networks that are very deep or have complex architec-
tures. Moreover, unlike ReLU, which outputs zero for any negative input, GELU
allows for a small, non-zero output for negative values. This characteristic means that
during the training process, neurons that receive negative input can still contribute to
the learning process, albeit to a lesser extent than positive inputs.
Next, let’s use the GELU function to implement the small neural network module,
FeedForward, that we will be using in the LLM’s transformer block later.

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
        GELU(),
        nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    def forward(self, x):
        return self.layers(x)
```
As we can see, the FeedForward module is a small neural network consisting of two
Linear layers and a GELU activation function. In the 124-million-parameter GPT
model, it receives the input batches with tokens that have an embedding size of 768
each via the GPT_CONFIG_124M dictionary where GPT_CONFIG_ 124M["emb_dim"] = 768.
Figure below shows how the embedding size is manipulated inside this small feed for-
ward neural network when we pass it some inputs.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image69.png?raw=true)

An overview of the connections between the layers of the feed forward neural network. This neural network can accommodate variable batch sizes and numbers of tokens in the input. However, the embedding size for each token is determined and fixed when initializing
the weights.

Following the example in figure 4.9, let’s initialize a new FeedForward module with a
token embedding size of 768 and feed it a batch input with two samples and three
tokens each:

```python
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)
```
As we can see, the shape of the output tensor is the same as that of the input tensor:
torch.Size([2, 3, 768])
The FeedForward module plays a crucial role in enhancing the model’s ability to learn
from and generalize the data. Although the input and output dimensions of this
module are the same, it internally expands the embedding dimension into a higher-
dimensional space through the first linear layer, as illustrated in figure below. This expan-
sion is followed by a nonlinear GELU activation and then a contraction back to the orig-
inal dimension with the second linear transformation. Such a design allows for the
exploration of a richer representation space.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image70.png?raw=true)

Moreover, the uniformity in input and output dimensions simplifies the architecture
by enabling the stacking of multiple layers, as we will do later, without the need to
adjust dimensions between them, thus making the model more scalable.
As figure below shows, we have now implemented most of the LLM’s building blocks.
Next, we will go over the concept of shortcut connections that we insert between dif-
ferent layers of a neural network, which are important for improving the training
performance in deep neural network architectures.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image71.png?raw=true)


## Adding shortcut connections
Let’s discuss the concept behind shortcut connections, also known as skip or residual
connections. Originally, shortcut connections were proposed for deep networks in
computer vision (specifically, in residual networks) to mitigate the challenge of van-
ishing gradients. The vanishing gradient problem refers to the issue where gradients
(which guide weight updates during training) become progressively smaller as they
propagate backward through the layers, making it difficult to effectively train earlier
layers.
Figure below shows that a shortcut connection creates an alternative, shorter path
for the gradient to flow through the network by skipping one or more layers, which is
achieved by adding the output of one layer to the output of a later layer. This is why
these connections are also known as skip connections. They play a crucial role in pre-
serving the flow of gradients during the backward pass in training.
In the following list, we implement the neural network in figure below to see how
we can add shortcut connections in the forward method.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image72.png?raw=true)

```python
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),
                GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),
                GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]),
                GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]),
                GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),
                GELU())
            ])
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x
```
The code implements a deep neural network with five layers, each consisting of a
Linear layer and a GELU activation function. In the forward pass, we iteratively pass the
input through the layers and optionally add the shortcut connections if the self.use_
shortcut attribute is set to True.
Let’s use this code to initialize a neural network without shortcut connections.
Each layer will be initialized such that it accepts an example with three input values
and returns three output values. The last layer returns a single output value:

```python
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(
layer_sizes, use_shortcut=False
)
```
Next, we implement a function that computes the gradients in the model’s back-
ward pass:
```python
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
```
This code specifies a loss function that computes how close the model output and a
user-specified target (here, for simplicity, the value 0) are. Then, when calling
loss.backward(), PyTorch computes the loss gradient for each layer in the model. We
can iterate through the weight parameters via model.named_parameters(). Suppose we
have a 3 × 3 weight parameter matrix for a given layer. In that case, this layer will have
3 × 3 gradient values, and we print the mean absolute gradient of these 3 × 3 gradient
values to obtain a single gradient value per layer to compare the gradients between
layers more easily.
In short, the .backward() method is a convenient method in PyTorch that com-
putes loss gradients, which are required during model training, without implement-
ing the math for the gradient calculation ourselves, thereby making working with
deep neural networks much more accessible.
Let’s now use the print_gradients function and apply it to the model without skip
connections:

print_gradients(model_without_shortcut, sample_input)

The output is
layers.0.0.weight has gradient mean of 0.00020173587836325169
layers.1.0.weight has gradient mean of 0.0001201116101583466
layers.2.0.weight has gradient mean of 0.0007152041653171182
layers.3.0.weight has gradient mean of 0.001398873864673078
layers.4.0.weight has gradient mean of 0.005049646366387606

The output of the print_gradients function shows, the gradients become smaller
as we progress from the last layer (layers.4) to the first layer (layers.0), which is
a phenomenon called the vanishing gradient problem.
Let’s now instantiate a model with skip connections and see how it compares:
```python
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)
```
The output is

layers.0.0.weight has gradient mean of 0.22169792652130127
layers.1.0.weight has gradient mean of 0.20694105327129364
layers.2.0.weight has gradient mean of 0.32896995544433594
layers.3.0.weight has gradient mean of 0.2665732502937317
layers.4.0.weight has gradient mean of 1.3258541822433472

The last layer (layers.4) still has a larger gradient than the other layers. However,
the gradient value stabilizes as we progress toward the first layer (layers.0) and
doesn’t shrink to a vanishingly small value.
In conclusion, shortcut connections are important for overcoming the limitations
posed by the vanishing gradient problem in deep neural networks. Shortcut connec-
tions are a core building block of very large models such as LLMs, and they will help
facilitate more effective training by ensuring consistent gradient flow across layers
when we train the GPT model in the next chapter.
Next, we’ll connect all of the previously covered concepts (layer normalization,
GELU activations, feed forward module, and shortcut connections) in a transformer
block, which is the final building block we need to code the GPT architecture.



## Connecting attention and linear layers in a transformer block
Now, let’s implement the transformer block, a fundamental building block of GPT and
other LLM architectures. This block, which is repeated a dozen times in the 124-million-
parameter GPT-2 architecture, combines several concepts we have previously covered:
multi-head attention, layer normalization, dropout, feed forward layers, and GELU
activations. Later, we will connect this transformer block to the remaining parts of the
GPT architecture.
Figure below shows a transformer block that combines several components, includ-
ing the masked multi-head attention module (see chapter 3) and the FeedForward
module we previously implemented (see section 4.3). When a transformer block pro-
cesses an input sequence, each element in the sequence (for example, a word or sub-
word token) is represented by a fixed-size vector (in this case, 768 dimensions). The
operations within the transformer block, including multi-head attention and feed for-
ward layers, are designed to transform these vectors in a way that preserves their
dimensionality.
The idea is that the self-attention mechanism in the multi-head attention block iden-
tifies and analyzes relationships between elements in the input sequence. In contrast,
the feed forward network modifies the data individually at each position. This combina-
tion not only enables a more nuanced understanding and processing of the input but
also enhances the model’s overall capacity for handling complex data patterns.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image73.png?raw=true)

An illustration of a transformer block. Input tokens have been embedded into 768-
dimensional vectors. Each row corresponds to one token’s vector representation. The outputs of the transformer block are vectors of the same dimension as the input, which can then be fed into
subsequent layers in an LLM.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image74.png?raw=true)
The given code defines a TransformerBlock class in PyTorch that includes a multi-head
attention mechanism (MultiHeadAttention) and a feed forward network (Feed-
Forward), both configured based on a provided configuration dictionary (cfg), such
as GPT_CONFIG_124M.
Layer normalization (LayerNorm) is applied before each of these two components,
and dropout is applied after them to regularize the model and prevent overfitting. This
is also known as Pre-LayerNorm. Older architectures, such as the original transformer
model, applied layer normalization after the self-attention and feed forward networks
instead, known as Post-LayerNorm, which often leads to worse training dynamics.
The class also implements the forward pass, where each component is followed by
a shortcut connection that adds the input of the block to its output. This critical fea-
ture helps gradients flow through the network during training and improves the
learning of deep models.
Using the GPT_CONFIG_124M dictionary we defined earlier, let’s instantiate a trans-
former block and feed it some sample data:

```python
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
```
The output is
Input shape: torch.Size([2, 4, 768])
Output shape: torch.Size([2, 4, 768])

As we can see, the transformer block maintains the input dimensions in its output, indi-
cating that the transformer architecture processes sequences of data without altering
their shape throughout the network.
The preservation of shape throughout the transformer block architecture is not
incidental but a crucial aspect of its design. This design enables its effective applica-
tion across a wide range of sequence-to-sequence tasks, where each output vector
directly corresponds to an input vector, maintaining a one-to-one relationship. How-
ever, the output is a context vector that encapsulates information from the entire
input sequence. This means that while the physical dimensions of the
sequence (length and feature size) remain unchanged as it passes through the trans-
former block, the content of each output vector is re-encoded to integrate contextual
information from across the entire input sequence.
With the transformer block implemented, we now have all the building blocks
needed to implement the GPT architecture. As illustrated in figure below, the trans-
former block combines layer normalization, the feed forward network, GELU activa-
tions, and shortcut connections. As we will eventually see, this transformer block will
make up the main component of the GPT architecture.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image75.png?raw=true)



## Coding the GPT model
We started this chapter with a big-picture overview of a GPT architecture that we
called DummyGPTModel. In this DummyGPTModel code implementation, we showed the
input and outputs to the GPT model, but its building blocks remained a black box
using a DummyTransformerBlock and DummyLayerNorm class as placeholders.
Let’s now replace the DummyTransformerBlock and DummyLayerNorm placeholders
with the real TransformerBlock and LayerNorm classes we coded previously to assem-
ble a fully working version of the original 124-million-parameter version of GPT-2. In
chapter 5, we will pretrain a GPT-2 model, and in chapter 6, we will load in the pre-
trained weights from OpenAI.
Before we assemble the GPT-2 model in code, let’s look at its overall structure, as
shown in figure below, which includes all the concepts we have covered so far. As we can
see, the transformer block is repeated many times throughout a GPT model architec-
ture. In the case of the 124-million-parameter GPT-2 model, it’s repeated 12 times,
which we specify via the n_layers entry in the GPT_CONFIG_124M dictionary. This
transform block is repeated 48 times in the largest GPT-2 model with 1,542 million
parameters.
The output from the final transformer block then goes through a final layer normal-
ization step before reaching the linear output layer. This layer maps the transformer’s
output to a high-dimensional space (in this case, 50,257 dimensions, corresponding to
the model’s vocabulary size) to predict the next token in the sequence.
Let’s now code the architecture in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image76.png?raw=true)
An overview of the GPT model architecture showing the flow of data through the GPT model.
Starting from the bottom, tokenized text is first converted into token embeddings, which are then augmented with positional embeddings. This combined information forms a tensor that is passed through a series of transformer blocks shown in the center (each containing multi-head attention and feed forward neural network layers with dropout and layer normalization), which are stacked on top of each other and repeated 12 times.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image77.png?raw=true)

Thanks to the TransformerBlock class, the GPTModel class is relatively small and
compact.
The __init__ constructor of this GPTModel class initializes the token and posi-
tional embedding layers using the configurations passed in via a Python dictionary,
cfg. These embedding layers are responsible for converting input token indices into
dense vectors and adding positional information (see chapter 2).
Next, the __init__ method creates a sequential stack of TransformerBlock mod-
ules equal to the number of layers specified in cfg. Following the transformer blocks,
a LayerNorm layer is applied, standardizing the outputs from the transformer blocks to
stabilize the learning process. Finally, a linear output head without bias is defined,
which projects the transformer’s output into the vocabulary space of the tokenizer to
generate logits for each token in the vocabulary.
The forward method takes a batch of input token indices, computes their embed-
dings, applies the positional embeddings, passes the sequence through the transformer
blocks, normalizes the final output, and then computes the logits, representing the next
token’s unnormalized probabilities. We will convert these logits into tokens and text
outputs in the next section.

Let’s now initialize the 124-million-parameter GPT model using the GPT_CONFIG_
124M dictionary we pass into the cfg parameter and feed it with the batch text input
we previously created:
```python
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)
```
This code prints the contents of the input batch followed by the output tensor:

![alt text](https://github.com/Rezashatery/LLM/blob/main/image78.png?raw=true)
As we can see, the output tensor has the shape [2, 4, 50257], since we passed in two
input texts with four tokens each. The last dimension, 50257, corresponds to the
vocabulary size of the tokenizer. Later, we will see how to convert each of these 50,257-
dimensional output vectors back into tokens.
Before we move on to coding the function that converts the model outputs into
text, let’s spend a bit more time with the model architecture itself and analyze its size.
Using the numel() method, short for “number of elements,” we can collect the total
number of parameters in the model’s parameter tensors:

```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
```
The result is
Total number of parameters: 163,009,536

Now, a curious reader might notice a discrepancy. Earlier, we spoke of initializing
a 124-million-parameter GPT model, so why is the actual number of parameters
163 million?
The reason is a concept called weight tying, which was used in the original GPT-2
architecture. It means that the original GPT-2 architecture reuses the weights from
the token embedding layer in its output layer. To understand better, let’s take a look at
the shapes of the token embedding layer and linear output layer that we initialized on
the model via the GPTModel earlier:

```python
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)
```
As we can see from the print outputs, the weight tensors for both these layers have the
same shape:

Token embedding layer shape: torch.Size([50257, 768])
Output layer shape: torch.Size([50257, 768])

The token embedding and output layers are very large due to the number of rows for
the 50,257 in the tokenizer’s vocabulary. Let’s remove the output layer parameter
count from the total GPT-2 model count according to the weight tying:

```python
total_params_gpt2 = (
total_params - sum(p.numel()
for p in model.out_head.parameters())
)
print(f"Number of trainable parameters "
f"considering weight tying: {total_params_gpt2:,}"
)
```

The output is
Number of trainable parameters considering weight tying: 124,412,160

As we can see, the model is now only 124 million parameters large, matching the orig-
inal size of the GPT-2 model.
Weight tying reduces the overall memory footprint and computational complexity
of the model. However, in my experience, using separate token embedding and out-
put layers results in better training and model performance; hence, we use separate
layers in our GPTModel implementation. The same is true for modern LLMs.
```python
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")
```
The result is
Total size of the model: 621.83 MB

In conclusion, by calculating the memory requirements for the 163 million parame-
ters in our GPTModel object and assuming each parameter is a 32-bit float taking up 4
bytes, we find that the total size of the model amounts to 621.83 MB, illustrating the
relatively large storage capacity required to accommodate even relatively small LLMs.
Now that we’ve implemented the GPTModel architecture and saw that it outputs
numeric tensors of shape [batch_size, num_tokens, vocab_size], let’s write the code
to convert these output tensors into text.



## Generating text
We will now implement the code that converts the tensor outputs of the GPT model
back into text. Before we get started, let’s briefly review how a generative model like
an LLM generates text one word (or token) at a time.
Figure below illustrates the step-by-step process by which a GPT model generates
text given an input context, such as “Hello, I am.” With each iteration, the input con-
text grows, allowing the model to generate coherent and contextually appropriate
text. By the sixth iteration, the model has constructed a complete sentence: “Hello, I
am a model ready to help.” We’ve seen that our current GPTModel implementation
outputs tensors with shape [batch_size, num_token, vocab_size]. Now the question
is: How does a GPT model go from these output tensors to the generated text?

![alt text](https://github.com/Rezashatery/LLM/blob/main/image79.png?raw=true)

The step-by-step process by which an LLM generates text, one
token at a time. Starting with an initial input context (“Hello, I am”), the
model predicts a subsequent token during each iteration, appending it to the
input context for the next round of prediction. As shown, the first iteration
adds “a,” the second “model,” and the third “ready,” progressively building
the sentence.

The process by which a GPT model goes from output tensors to generated text
involves several steps, as illustrated in figure below. These steps include decoding the
output tensors, selecting tokens based on a probability distribution, and converting
these tokens into human-readable text.
The next-token generation process detailed in figure below illustrates a single step
where the GPT model generates the next token given its input. In each step, the model
outputs a matrix with vectors representing potential next tokens. The vector corre-
sponding to the next token is extracted and converted into a probability distribution via
the softmax function. Within the vector containing the resulting probability scores, the
index of the highest value is located, which translates to the token ID. This token ID is
then decoded back into text, producing the next token in the sequence. Finally, this
token is appended to the previous inputs, forming a new input sequence for the subse-
quent iteration. This step-by-step process enables the model to generate text sequen-
tially, building coherent phrases and sentences from the initial input context.
In practice, we repeat this process over many iterations, such as shown in figure above,
until we reach a user-specified number of generated tokens. In code, we can imple-
ment the token-generation process as shown after the figure.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image80.png?raw=true)

![alt text](https://github.com/Rezashatery/LLM/blob/main/image81.png?raw=true)

This code demonstrates a simple implementation of a generative loop for a lan-
guage model using PyTorch. It iterates for a specified number of new tokens to be
generated, crops the current context to fit the model’s maximum context size, com-
putes predictions, and then selects the next token based on the highest probability
prediction.
To code the generate_text_simple function, we use a softmax function to con-
vert the logits into a probability distribution from which we identify the position with
the highest value via torch.argmax. The softmax function is monotonic, meaning it
preserves the order of its inputs when transformed into outputs. So, in practice, the
softmax step is redundant since the position with the highest score in the softmax out-
put tensor is the same position in the logit tensor. In other words, we could apply the
torch.argmax function to the logits tensor directly and get identical results. However,
I provide the code for the conversion to illustrate the full process of transforming log-
its to probabilities, which can add additional intuition so that the model generates the
most likely next token, which is known as greedy decoding.
When we implement the GPT training code in the next chapter, we will use addi-
tional sampling techniques to modify the softmax outputs such that the model doesn’t
always select the most likely token. This introduces variability and creativity in the gen-
erated text.
This process of generating one token ID at a time and appending it to the context
using the generate_text_simple function is further illustrated in figure below.  We generate
the token IDs in an iterative fashion. For instance, in iteration 1, the model is pro-
vided with the tokens corresponding to “Hello, I am,” predicts the next token (with
ID 257, which is “a”), and appends it to the input. This process is repeated until the
model produces the complete sentence “Hello, I am a model ready to help” after six
iterations.
Let’s now try out the generate_text_simple function with the "Hello, I am" con-
text as model input. First, we encode the input context into token IDs:

```python
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) #Adds batch dimension
print("encoded_tensor.shape:", encoded_tensor.shape)
```
The encoded IDs are
encoded: [15496, 11, 314, 716]
encoded_tensor.shape: torch.Size([1, 4])

![alt text](https://github.com/Rezashatery/LLM/blob/main/image82.png?raw=true)

The six iterations of a token prediction cycle, where the model takes a sequence of initial token IDs as input, predicts the next token, and appends this token to the input sequence for the next iteration. (The token IDs are also translated into their corresponding text for better understanding.)

Next, we put the model into .eval() mode. This disables random components like
dropout, which are only used during training, and use the generate_text_simple
function on the encoded input tensor:
```python
model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))
```
The resulting output token IDs are

Output: tensor([[15496,11,314,716,27018,24086,47843,30961,42348,7267]])
Output length: 10

Using the .decode method of the tokenizer, we can convert the IDs back into text:
```python
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
```
The model output in text format is
Hello, I am Featureiman Byeswickattribute argue

As we can see, the model generated gibberish, which is not at all like the coherent text
Hello, I am a model ready to help. What happened? The reason the model is unable to
produce coherent text is that we haven’t trained it yet. So far, we have only implemented
the GPT architecture and initialized a GPT model instance with initial random weights.
Model training is a large topic in itself, and we will tackle it in the next chapter.



# Petraining on unlabeled data (CHAPTER 5)
Thus far, we have implemented the data sampling and attention mechanism and
coded the LLM architecture. It is now time to implement a training function and
pretrain the LLM. We will learn about basic model evaluation techniques to mea-
sure the quality of the generated text, which is a requirement for optimizing the
LLM during the training process. Moreover, we will discuss how to load pretrained
weights, giving our LLM a solid starting point for fine-tuning. Figure below lays out
our overall plan, highlighting what we will discuss in this chapter.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image83.png?raw=true)

**Weight parameters**

In the context of LLMs and other deep learning models, weights refer to the trainable
parameters that the learning process adjusts. These weights are also known as
weight parameters or simply parameters. In frameworks like PyTorch, these weights
are stored in linear layers; we used these to implement the multi-head attention mod-
ule in chapter 3 and the GPTModel in chapter 4. After initializing a layer (new_layer
= torch.nn.Linear(...)), we can access its weights through the .weight attri-
bute, new_layer.weight. Additionally, for convenience, PyTorch allows direct
access to all a model’s trainable parameters, including weights and biases, through
the method model.parameters(), which we will use later when implementing the
model training.

## Evaluating generative text models
After briefly recapping the text generation from chapter 4, we will set up our LLM for
text generation and then discuss basic ways to evaluate the quality of the generated text.
We will then calculate the training and validation losses. Figure below shows the topics
covered in this chapter, with these first three steps highlighted.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image84.png?raw=true)

## Using GPT to generate text
Let’s set up the LLM and briefly recap the text generation process we implemented in
chapter 4. We begin by initializing the GPT model that we will later evaluate and train
using the GPTModel class and GPT_CONFIG_124M dictionary.

```python
import torch
from chapter04 import GPTModel
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, # We shorten the context length from 1,024 to 256 tokens.
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1, # It’s possible and common to set dropout to 0.
    "qkv_bias": False
}
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()
```
Considering the GPT_CONFIG_124M dictionary, the only adjustment we have made com-
pared to the previous chapter is that we have reduced the context length (context_
length) to 256 tokens. This modification reduces the computational demands of
training the model, making it possible to carry out the training on a standard laptop
computer.
Originally, the GPT-2 model with 124 million parameters was configured to handle
up to 1,024 tokens. After the training process, we will update the context size setting
and load pretrained weights to work with a model configured for a 1,024-token con-
text length.
Using the GPTModel instance, we adopt the generate_text_simple function from
chapter 4 and introduce two handy functions: text_to_token_ ids and token_ids_
to_text. These functions facilitate the conversion between text and token represen-
tations, a technique we will utilize throughout this chapter.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image85.png?raw=true)

Figure above illustrates a three-step text generation process using a GPT model. First,
the tokenizer converts input text into a series of token IDs (see chapter 2). Second,
the model receives these token IDs and generates corresponding logits, which are vec-
tors representing the probability distribution for each token in the vocabulary (see
chapter 4). Third, these logits are converted back into token IDs, which the tokenizer
decodes into human-readable text, completing the cycle from textual input to tex-
tual output.

We can implement the text generation process, as shown in the following code.

```python
import tiktoken
from chapter04 import generate_text_simple
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())   
start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
) 
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```
Using this code, the model generates the following text:
Output text:
Every effort moves you rentingetic wasn? refres RexMeCHicular stren

Clearly, the model isn’t yet producing coherent text because it hasn’t undergone
training. To define what makes text “coherent” or “high quality,” we have to imple-
ment a numerical method to evaluate the generated content. This approach will
enable us to monitor and enhance the model’s performance throughout its training
process.


## Calculating the text generation loss
Next, let’s explore techniques for numerically assessing text quality generated
during training by calculating a text generation loss. We will go over this topic step by
step with a practical example to make the concepts clear and applicable, beginning
with a short recap of how the data is loaded and how the text is generated via the
generate_text_simple function.
Next, let’s explore techniques for numerically assessing text quality generated
during training by calculating a text generation loss. We will go over this topic step by
step with a practical example to make the concepts clear and applicable, beginning
with a short recap of how the data is loaded and how the text is generated via the
generate_text_simple function.
Figure below illustrates the overall flow from input text to LLM-generated text using a
five-step procedure. This text-generation process shows what the generate_text_simple
function does internally. We need to perform these same initial steps before we can
compute a loss that measures the generated text quality later in this section.
Figure below outlines the text generation process with a small seven-token vocabulary
to fit this image on a single page. However, our GPTModel works with a much larger
vocabulary consisting of 50,257 words; hence, the token IDs in the following code will
range from 0 to 50,256 rather than 0 to 6.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image86.png?raw=true)

For each of the three input tokens, shown on the left, we compute a vector containing probability scores corresponding to each token in the vocabulary. The index position of the highest probability score in each vector represents the most likely next token ID. These token IDs associated with the highest probability scores are selected and mapped back into a text that represents the text generated by the model.
Also, figure above only shows a single text example ("every effort moves") for sim-
plicity. In the following hands-on code example that implements the steps in the fig-
ure, we will work with two input examples for the GPT model ("every effort moves"
and "I really like").

Consider these two input examples, which have already been mapped to token IDs
(figure above, step 1):
```python
inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
                        [40,1107, 588]])    # "I really like"]
```
Matching these inputs, the targets contain the token IDs we want the model to
produce:

```python
targets = torch.tensor([[3626, 6100, 345 ],     # [" effort moves you",
                        [1107, 588, 11311]])    # # " really like chocolate"]
```
Note that the targets are the inputs but shifted one position forward, a concept we
covered in chapter 2 during the implementation of the data loader. This shifting strat-
egy is crucial for teaching the model to predict the next token in a sequence.

Now we feed the inputs into the model to calculate logits vectors for the two input
examples, each comprising three tokens. Then we apply the softmax function to
transform these logits into probability scores(step 2).
```python
with torch.no_grad():   # Disables gradient tracking since we are not training yet
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1) #Probability of each token in vocabulary
print(probas.shape)
```
The resulting tensor dimension of the probability score (probas) tensor is
torch.Size([2, 3, 50257])

The first number, 2, corresponds to the two examples (rows) in the inputs, also known
as batch size. The second number, 3, corresponds to the number of tokens in each
input (row). Finally, the last number corresponds to the embedding dimensionality,
which is determined by the vocabulary size. Following the conversion from logits to
probabilities via the softmax function, the generate_text_simple function then con-
verts the resulting probability scores back into text (figure above, steps 3–5).
We can complete steps 3 and 4 by applying the argmax function to the probability
scores to obtain the corresponding token IDs:
```python
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)
```
Given that we have two input batches, each containing three tokens, applying the
argmax function to the probability scores (figure above, step 3) yields two sets of outputs,
each with three predicted token IDs:
```python
Token IDs:
tensor([[[16657],
[ 339],
[42826]],
[[49906], # second batch
[29669],
[41751]]])
```
Finally, step 5 converts the token IDs back into text:

```python
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:"
f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
```
When we decode these tokens, we find that these output tokens are quite different
from the target tokens we want the model to generate:
Targets batch 1: effort moves you
Outputs batch 1: Armed heNetflix

The model produces random text that is different from the target text because it has
not been trained yet. We now want to evaluate the performance of the model’s gen-
erated text numerically via a loss (figure below). Not only is this useful for measuring
the quality of the generated text, but it’s also a building block for implementing the
training function, which we will use to update the model’s weight to improve the
generated text.


![alt text](https://github.com/Rezashatery/LLM/blob/main/image87.png?raw=true)

Part of the text evaluation process that we implement, as shown in figure above, is to mea-
sure “how far” the generated tokens are from the correct predictions (targets). The
training function we implement later will use this information to adjust the model
weights to generate text that is more similar to (or, ideally, matches) the target text.
The model training aims to increase the softmax probability in the index positions
corresponding to the correct target token IDs, as illustrated in figure below. This softmax
probability is also used in the evaluation metric we will implement next to numerically
assess the model’s generated outputs: the higher the probability in the correct posi-
tions, the better.
Remember that figure below displays the softmax probabilities for a compact seven-
token vocabulary to fit everything into a single figure. This implies that the starting
random values will hover around 1/7, which equals approximately 0.14. However, the
vocabulary we are using for our GPT-2 model has 50,257 tokens, so most of the initial
probabilities will hover around 0.00002 (1/50,257).

![alt text](https://github.com/Rezashatery/LLM/blob/main/image88.png?raw=true)

For each of the two input texts, we can print the initial softmax probability scores cor-
responding to the target tokens using the following code:

```python
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)
text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)
```
The three target token ID probabilities for each batch are
Text 1: tensor([7.4541e-05, 3.1061e-05, 1.1563e-05])
Text 2: tensor([1.0337e-05, 5.6776e-05, 4.7559e-06])

The goal of training an LLM is to maximize the likelihood of the correct token, which
involves increasing its probability relative to other tokens. This way, we ensure the
LLM consistently picks the target token—essentially the next word in the sentence—
as the next token it generates.

**Backpropagation**

How do we maximize the softmax probability values corresponding to the target
tokens? The big picture is that we update the model weights so that the model outputs
higher values for the respective token IDs we want to generate. The weight update is
done via a process called backpropagation, a standard technique for training deep
neural networks.
Backpropagation requires a loss function, which calculates the difference between
the model’s predicted output (here, the probabilities corresponding to the target
token IDs) and the actual desired output. This loss function measures how far off the
model’s predictions are from the target values.

Next, we will calculate the loss for the probability scores of the two example batches,
target_probas_1 and target_probas_2. The main steps are illustrated in figure below.
Since we already applied steps 1 to 3 to obtain target_probas_1 and target_
probas_2, we proceed with step 4, applying the logarithm to the probability scores:

```python
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
```
![alt text](https://github.com/Rezashatery/LLM/blob/main/image89.png?raw=true)

Calculating the loss involves several steps. Steps 1 to 3, which we have already
completed, calculate the token probabilities corresponding to the target tensors. These
probabilities are then transformed via a logarithm and averaged in steps 4 to 6.

This results in the following values:
tensor([ -9.5042, -10.3796, -11.3677, -11.4798,-9.7764, -12.2561])

Working with logarithms of probability scores is more manageable in mathematical
optimization than handling the scores directly. 
Next, we combine these log probabilities into a single score by computing the average.

```python
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
```
The resulting average log probability score is
tensor(-10.7940)

The goal is to get the average log probability as close to 0 as possible by updating the
model’s weights as part of the training process. However, in deep learning, the com-
mon practice isn’t to push the average log probability up to 0 but rather to bring the
negative average log probability down to 0. The negative average log probability is
simply the average log probability multiplied by –1, which corresponds to step 6.

```python
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)
```
This prints tensor(10.7940). In deep learning, the term for turning this negative
value, –10.7940, into 10.7940, is known as the cross entropy loss. PyTorch comes in
handy here, as it already has a built-in cross_entropy function that takes care of all
these six steps.



**Cross entropy loss**

At its core, the cross entropy loss is a popular measure in machine learning and deep
learning that measures the difference between two probability distributions—typi-
cally, the true distribution of labels (here, tokens in a dataset) and the predicted dis-
tribution from a model (for instance, the token probabilities generated by an LLM).
In the context of machine learning and specifically in frameworks like PyTorch, the
cross_entropy function computes this measure for discrete outcomes, which is
similar to the negative average log probability of the target tokens given the model’s
generated token probabilities, making the terms “cross entropy” and “negative aver-
age log probability” related and often used interchangeably in practice.

Before we apply the cross_entropy function, let’s briefly recall the shape of the logits
and target tensors:
```python
print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)
```
The resulting shapes are
Logits shape: torch.Size([2, 3, 50257])
Targets shape: torch.Size([2, 3])

As we can see, the logits tensor has three dimensions: batch size, number of tokens,
and vocabulary size. The targets tensor has two dimensions: batch size and number
of tokens.
For the cross_entropy loss function in PyTorch, we want to flatten these tensors
by combining them over the batch dimension:

```python
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)
```
The resulting tensor dimensions are
Flattened logits: torch.Size([6, 50257])
Flattened targets: torch.Size([6])

The resulting tensor dimensions are
Flattened logits: torch.Size([6, 50257])
Flattened targets: torch.Size([6])

Remember that the targets are the token IDs we want the LLM to generate, and the
logits contain the unscaled model outputs before they enter the softmax function to
obtain the probability scores.
Previously, we applied the softmax function, selected the probability scores corre-
sponding to the target IDs, and computed the negative average log probabilities.
PyTorch’s cross_entropy function will take care of all these steps for us:

```python
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)
```
The resulting loss is the same that we obtained previously when applying the individ-
ual steps in figure above manually:
tensor(10.7940)

**Perplexity**

Perplexity is a measure often used alongside cross entropy loss to evaluate the per-
formance of models in tasks like language modeling. It can provide a more interpre-
table way to understand the uncertainty of a model in predicting the next token in a
sequence.
Perplexity measures how well the probability distribution predicted by the model
matches the actual distribution of the words in the dataset. Similar to the loss, a lower
perplexity indicates that the model predictions are closer to the actual distribution.
Perplexity can be calculated as perplexity = torch.exp(loss), which returns
tensor(48725.8203) when applied to the previously calculated loss.
Perplexity is often considered more interpretable than the raw loss value because it sig-
nifies the effective vocabulary size about which the model is uncertain at each step. In
the given example, this would translate to the model being unsure about which among
48,725 tokens in the vocabulary to generate as the next token.

We have now calculated the loss for two small text inputs for illustration purposes.
Next, we will apply the loss computation to the entire training and validation sets.




## Calculating the training and validation set losses
We must first prepare the training and validation datasets that we will use to train the
LLM. Then, as highlighted in figure below, we will calculate the cross entropy for the
training and validation sets, which is an important component of the model training
process.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image90.png?raw=true)

Having completed steps 1 and 2, including computing the cross entropy loss, we can now apply this loss computation to the entire text dataset that we will use for model training.

To compute the loss on the training and validation datasets, we use a very small text
dataset, the “The Verdict” short story by Edith Wharton, which we have already
worked with in chapter 2. By selecting a text from the public domain, we circumvent
any concerns related to usage rights. Additionally, using such a small dataset allows
for the execution of code examples on a standard laptop computer in a matter of minutes, even without a high-end GPU, which is particularly advantageous for edu-
cational purposes.

**The cost of pretraining LLMs**

To put the scale of our project into perspective, consider the training of the 7 billion
parameter Llama 2 model, a relatively popular openly available LLM. This model
required 184,320 GPU hours on expensive A100 GPUs, processing 2 trillion tokens.
At the time of writing, running an 8 × A100 cloud server on AWS costs around $30
per hour. A rough estimate puts the total training cost of such an LLM at around
$690,000 (calculated as 184,320 hours divided by 8, then multiplied by $30).

The following code loads the “The Verdict” short story:
```python
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
text_data = file.read()
```
After loading the dataset, we can check the number of characters and tokens in the
dataset:

```python
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)
```
The output is
Characters: 20479
Tokens: 5145

With just 5,145 tokens, the text might seem too small to train an LLM, but as men-
tioned earlier, it’s for educational purposes so that we can run the code in minutes
instead of weeks. Plus, later we will load pretrained weights from OpenAI into our
GPTModel code.
Next, we divide the dataset into a training and a validation set and use the data
loaders from chapter 2 to prepare the batches for LLM training. This process is visual-
ized in figure below. Due to spatial constraints, we use a max_length=6. However, for the
actual data loaders, we set the max_length equal to the 256-token context length that
the LLM supports so that the LLM sees longer texts during training.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image91.png?raw=true)

When preparing the data loaders, we split the input text into training and validation set portions. Then we tokenize the text (only shown for the training set portion for simplicity) and divide the tokenized text into chunks of a user-specified length (here, 6). Finally, we shuffle the rows and organize the chunked text into batches (here, batch size 2), which we can use for model training.

We are training the model with training data presented in similarly
sized chunks for simplicity and efficiency. However, in practice, it can also be
beneficial to train an LLM with variable-length inputs to help the LLM to bet-
ter generalize across different types of inputs when it is being used.

To implement the data splitting and loading, we first define a train_ratio to use 90%
of the data for training and the remaining 10% as validation data for model evalua-
tion during training:
```python
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
```
Using the train_data and val_data subsets, we can now create the respective data
loader reusing the create_dataloader_v1 code from chapter 2:

```python
from chapter02 import create_dataloader_v1
torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)
```

We used a relatively small batch size to reduce the computational resource demand
because we were working with a very small dataset. In practice, training LLMs with
batch sizes of 1,024 or larger is not uncommon.
As an optional check, we can iterate through the data loaders to ensure that they
were created correctly:
```python
print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)
print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)
```

We should see the following outputs:
Train loader:
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])

Validation loader:
torch.Size([2, 256]) torch.Size([2, 256])

Based on the preceding code output, we have nine training set batches with two sam-
ples and 256 tokens each. Since we allocated only 10% of the data for validation, there
is only one validation batch consisting of two input examples. As expected, the input
data (x) and target data (y) have the same shape (the batch size times the number of
tokens in each batch) since the targets are the inputs shifted by one position, as dis-
cussed in chapter 2.

Next, we implement a utility function to calculate the cross entropy loss of a given
batch returned via the training and validation loader:

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss
```
We can now use this calc_loss_batch utility function, which computes the loss for a
single batch, to implement the following calc_loss_loader function that computes
the loss over all the batches sampled by a given data loader.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image92.png?raw=true)

By default, the calc_loss_loader function iterates over all batches in a given data
loader, accumulates the loss in the total_loss variable, and then computes and
averages the loss over the total number of batches. Alternatively, we can specify a
smaller number of batches via num_batches to speed up the evaluation during model
training.
Let’s now see this calc_loss_loader function in action, applying it to the training
and validation set loaders:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) #If you have a machine with a CUDA-supported GPU, the LLM will train on the GPU without making any changes to the code.
with torch.no_grad(): #Disables gradient tracking for efficiency because we are not training yet
    train_loss = calc_loss_loader(train_loader, model, device) 
    # Via the “device” setting,we ensure the data is loaded onto the same device as the LLM model.
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

```
The resulting loss values are
Training loss: 10.98758347829183
Validation loss: 10.98110580444336

The loss values are relatively high because the model has not yet been trained. For
comparison, the loss approaches 0 if the model learns to generate the next tokens as
they appear in the training and validation sets.
Now that we have a way to measure the quality of the generated text, we will train
the LLM to reduce this loss so that it becomes better at generating text, as illustrated
in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image93.png?raw=true)

Next, we will focus on pretraining the LLM. After model training, we will implement
alternative text generation strategies and save and load pretrained model weights.



## Training an LLM
It is finally time to implement the code for pretraining the LLM, our GPTModel. For this,
we focus on a straightforward training loop to keep the code concise and readable.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image94.png?raw=true)

A typical training loop for training deep neural networks in PyTorch consists of numerous steps, iterating over the batches in the training set for several epochs. In each loop, we calculate the loss for each training set batch to determine loss gradients, which we use to update the model weights so that the training set loss is minimized.

The flowchart in figure above depicts a typical PyTorch neural network training work-
flow, which we use for training an LLM. It outlines eight steps, starting with iterating
over each epoch, processing batches, resetting gradients, calculating the loss and new gradients, and updating weights and concluding with monitoring steps like printing
losses and generating text samples.

We can implement this training flow via the train_model_simple function in code.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image95.png?raw=true)

Note that the train_model_simple function we just created uses two functions we
have not defined yet: evaluate_model and generate_and_print_sample.
The evaluate_model function corresponds to step 7 in figure above. It prints the
training and validation set losses after each model update so we can evaluate whether
the training improves the model. More specifically, the evaluate_model function cal-
culates the loss over the training and validation set while ensuring the model is in eval-
uation mode with gradient tracking and dropout disabled when calculating the loss
over the training and validation sets:

```python
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # Dropout is disabled during evaluation for stable, reproducible results.
    with torch.no_grad(): 
    #Disables gradient tracking, which is not required during evaluation, to reduce
the computational overhead
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss
```

Similar to evaluate_model, the generate_and_print_sample function is a convenience
function that we use to track whether the model improves during the training. In partic-
ular, the generate_and_print_sample function takes a text snippet (start_context) as
input, converts it into token IDs, and feeds it to the LLM to generate a text sample
using the generate_text_simple function we used earlier:
```python
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
        model=model, idx=encoded,
        max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
```
While the evaluate_model function gives us a numeric estimate of the model’s train-
ing progress, this generate_and_print_sample text function provides a concrete text
example generated by the model to judge its capabilities during training.

**AdamW**

Adam optimizers are a popular choice for training deep neural networks. However, in
our training loop, we opt for the AdamW optimizer. AdamW is a variant of Adam that
improves the weight decay approach, which aims to minimize model complexity and
prevent overfitting by penalizing larger weights. This adjustment allows AdamW to
achieve more effective regularization and better generalization; thus, AdamW is fre-
quently used in the training of LLMs.

Let’s see this all in action by training a GPTModel instance for 10 epochs using an
AdamW optimizer and the train_model_simple function we defined earlier:
```python
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)
```
Executing the train_model_simple function starts the training process, which takes
about 5 minutes to complete on a MacBook Air or a similar laptop. The output
printed during this execution is as follows:
```python
Ep 1 (Step 000000): Train loss 9.781, Val loss 9.933
Ep 1 (Step 000005): Train loss 8.111, Val loss 8.339
Every effort moves you,,,,,,,,,,,,.
Ep 2 (Step 000010): Train loss 6.661, Val loss 7.048
Ep 2 (Step 000015): Train loss 5.961, Val loss 6.616
Every effort moves you, and, and, and, and, and, and, and, and, and, and,
and, and, and, and, and, and, and, and, and, and, and, and,, and, and,
[...]
Ep 9 (Step 000080): Train loss 0.541, Val loss 6.393
Every effort moves you?" "Yes--quite insensible to the irony. She wanted
him vindicated--and by me!" He laughed again, and threw back the
window-curtains, I had the donkey. "There were days when I
Ep 10 (Step 000085): Train loss 0.391, Val loss 6.452
Every effort moves you know," was one of the axioms he laid down across the
Sevres and silver of an exquisitely appointed luncheon-table, when, on a
later day, I had again run over from Monte Carlo; and Mrs. Gis
```
As we can see, the training loss improves drastically, starting with a value of 9.781
and converging to 0.391. The language skills of the model have improved quite a
lot. In the beginning, the model is only able to append commas to the start context
(Every effort moves you,,,,,,,,,,,,) or repeat the word and. At the end of the
training, it can generate grammatically correct text.
Similar to the training set loss, we can see that the validation loss starts high
(9.933) and decreases during the training. However, it never becomes as small as the
training set loss and remains at 6.452 after the 10th epoch.
Before discussing the validation loss in more detail, let’s create a simple plot that
shows the training and validation set losses side by side:

```python
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
```
The resulting training and validation loss plot is shown in figure below. As we can see,
both the training and validation losses start to improve for the first epoch. However,
the losses start to diverge past the second epoch. This divergence and the fact that the
validation loss is much larger than the training loss indicate that the model is overfit-
ting to the training data. We can confirm that the model memorizes the training data
verbatim by searching for the generated text snippets, such as quite insensible to
the irony in the “The Verdict” text file.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image96.png?raw=true)

This memorization is expected since we are working with a very, very small training
dataset and training the model for multiple epochs. Usually, it’s common to train a
model on a much larger dataset for only one epoch.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image97.png?raw=true)

As illustrated in figure 5.13, we have completed four of our objectives for this chaper.
Next, we will cover text generation strategies for LLMs to reduce training data memo-
rization and increase the originality of the LLM-generated text before we cover weight
loading and saving and loading pretrained weights from OpenAI’s GPT model.

## Decoding strategies to control randomness

We begin by transferring the model back from the GPU to the CPU since infer-
ence with a relatively small model does not require a GPU. Also, after training, we put
the model into evaluation mode to turn off random components such as dropout:
model.to("cpu")
model.eval()

Next, we plug the GPTModel instance (model) into the generate_text_simple func-
tion, which uses the LLM to generate one token at a time:

```python
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```

The generated text is
Output text:
Every effort moves you know," was one of the axioms he laid down across the
Sevres and silver of an exquisitely appointed lun

As explained earlier, the generated token is selected at each generation step corre-
sponding to the largest probability score among all tokens in the vocabulary. This
means that the LLM will always generate the same outputs even if we run the preced-
ing generate_text_simple function multiple times on the same start context (Every
effort moves you).

## Temperature scaling
Let’s now look at temperature scaling, a technique that adds a probabilistic selection
process to the next-token generation task. Previously, inside the generate_text_simple
function, we always sampled the token with the highest probability as the next token
using torch.argmax, also known as greedy decoding. To generate text with more variety,
we can replace argmax with a function that samples from a probability distribution
(here, the probability scores the LLM generates for each vocabulary entry at each
token generation step).
To illustrate the probabilistic sampling with a concrete example, let’s briefly dis-
cuss the next-token generation process using a very small vocabulary for illustration
purposes:
```python
vocab = {
"closer": 0,
"every": 1,
"effort": 2,
"forward": 3,
"inches": 4,
"moves": 5,
"pizza": 6,
"toward": 7,
"you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}

```

Next, assume the LLM is given the start context "every effort moves you" and gener-
ates the following next-token logits:

next_token_logits = torch.tensor(
[4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

As discussed in chapter 4, inside generate_text_simple, we convert the logits into
probabilities via the softmax function and obtain the token ID corresponding to the
generated token via the argmax function, which we can then map back into text via
the inverse vocabulary:

```python
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
print(inverse_vocab[next_token_id])
```
Since the largest logit value and, correspondingly, the largest softmax probability
score are in the fourth position (index position 3 since Python uses 0 indexing), the
generated word is "forward".
To implement a probabilistic sampling process, we can now replace argmax with
the multinomial function in PyTorch:

```python
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])
```
The printed output is "forward" just like before. What happened? The multinomial
function samples the next token proportional to its probability score.

We can further control the distribution and selection process via a concept called
temperature scaling. Temperature scaling is just a fancy description for dividing the logits
by a number greater than 0:
```python
def softmax_with_temperature(logits, temperature):
scaled_logits = logits / temperature
return torch.softmax(scaled_logits, dim=0)
```
Temperatures greater than 1 result in more uniformly distributed token probabilities,
and temperatures smaller than 1 will result in more confident (sharper or more peaky)
distributions. Let’s illustrate this by plotting the original probabilities alongside proba-
bilities scaled with different temperature values:
```python
temperatures = [1, 0.1, 5]
scaled_probas = [softmax_with_temperature(next_token_logits, T)
                for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
         = ax.bar(x + i * bar_width, scaled_probas[i],
                bar_width, label=f'Temperature = {T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()
```

The resulting plot is shown in figure below.
A temperature of 1 divides the logits by 1 before passing them to the softmax func-
tion to compute the probability scores. In other words, using a temperature of 1 is the
same as not using any temperature scaling. In this case, the tokens are selected with a
probability equal to the original softmax probability scores via the multinomial sam-
pling function in PyTorch. For example, for the temperature setting 1, the token cor-
responding to “forward” would be selected about 60% of the time, as we can see in
figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image98.png?raw=true)

A temperature of 1 represents the unscaled probability
scores for each token in the vocabulary. Decreasing the temperature to
0.1 sharpens the distribution, so the most likely token (here, “forward”)
will have an even higher probability score. Likewise, increasing the
temperature to 5 makes the distribution more uniform.
This can add more
variety to the generated texts but also more often results in nonsensical text. For
example, using the temperature of 5 results in texts such as every effort moves you
pizza about 4% of the time.


## Top-k sampling
We’ve now implemented a probabilistic sampling approach coupled with temperature
scaling to increase the diversity of the outputs. We saw that higher temperature values
result in more uniformly distributed next-token probabilities, which result in more
diverse outputs as it reduces the likelihood of the model repeatedly selecting the most
probable token. This method allows for the exploring of less likely but potentially
more interesting and creative paths in the generation process. However, one down-
side of this approach is that it sometimes leads to grammatically incorrect or com-
pletely nonsensical outputs such as every effort moves you pizza.
Top-k sampling, when combined with probabilistic sampling and temperature scal-
ing, can improve the text generation results. In top-k sampling, we can restrict the
sampled tokens to the top-k most likely tokens and exclude all other tokens from the
selection process by masking their probability scores, as illustrated in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image99.png?raw=true)
Using top-k sampling with k = 3, we focus on the three tokens associated with the highest logits
and mask out all other tokens with negative infinity (–inf) before applying the softmax function. This results in a probability distribution with a probability value 0 assigned to all non-top-k tokens. (The numbers in this figure are truncated to two digits after the decimal point to reduce visual clutter. The values in the “Softmax” row should add up to 1.0.)

In code, we can implement the top-k procedure in figure above as follows, starting
with the selection of the tokens with the largest logit values:

```python
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top logits:", top_logits)
print("Top positions:", top_pos)
```
The logits values and token IDs of the top three tokens, in descending order, are
Top logits: tensor([6.7500, 6.2800, 4.5100])
Top positions: tensor([3, 7, 0])

Subsequently, we apply PyTorch’s where function to set the logit values of tokens that are
below the lowest logit value within our top-three selection to negative infinity (-inf):
```python
new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],  # Identifies logits less than the minimum in the top 3
    input=torch.tensor(float('-inf')), #Assigns –inf to these lower logits
    other=next_token_logits # Retains the original logits for all other tokens
)
print(new_logits)
```

The resulting logits for the next token in the nine-token vocabulary are

tensor([4.5100, -inf, -inf, 6.7500, -inf, -inf,-inf, 6.2800,-inf])

Lastly, let’s apply the softmax function to turn these into next-token probabilities:

```python
topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)
```
As we can see, the result of this top-three approach are three non-zero probability
scores:
tensor([0.0615, 0.0000, 0.0000, 0.5775, 0.0000, 0.0000, 0.0000, 0.3610,0.0000])

We can now apply the temperature scaling and multinomial function for probabilistic
sampling to select the next token among these three non-zero probability scores to
generate the next token. We do this next by modifying the text generation function.


## Modifying the text generation function
Now, let’s combine temperature sampling and top-k sampling to modify the generate_
text_simple function we used to generate text via the LLM earlier, creating a new
generate function.

```python
def generate(model, idx, max_new_tokens, context_size,
            temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
l           logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
        top_logits, _ = torch.topk(logits, top_k)
        min_val = top_logits[:, -1]
        logits = torch.where(
            logits < min_val,
            torch.tensor(float('-inf')).to(logits.device),
            logits
        )
        if temperature > 0.0:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        else:
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
        break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx            
```
Let’s now see this new generate function in action:

```python
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```
The generated text is
Output text:
Every effort moves you stand to work on surprise, a one of us had gone
with random-

As we can see, the generated text is very different from the one we previously gener-
ated via the generate_simple function ("Every effort moves you know,"
was one of the axioms he laid...! ), which was a memorized passage from the train-
ing set.

## Loading and saving model weights in PyTorch
Thus far, we have discussed how to numerically evaluate the training progress and pre-
train an LLM from scratch. Even though both the LLM and dataset were relatively
small, this exercise showed that pretraining LLMs is computationally expensive. Thus,
it is important to be able to save the LLM so that we don’t have to rerun the training
every time we want to use it in a new session.
Later, we will load a more capable pretrained GPT model from OpenAI into
our GPTModel instance.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image100.png?raw=true)

Fortunately, saving a PyTorch model is relatively straightforward. The recommended
way is to save a model’s state_dict, a dictionary mapping each layer to its parameters,
using the torch.save function:
torch.save(model.state_dict(), "model.pth")

"model.pth" is the filename where the state_dict is saved. The .pth extension is a
convention for PyTorch files, though we could technically use any file extension.
Then, after saving the model weights via the state_dict, we can load the model
weights into a new GPTModel model instance:
```python
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
```
As discussed in chapter 4, dropout helps prevent the model from overfitting to the
training data by randomly “dropping out” of a layer’s neurons during training. How-
ever, during inference, we don’t want to randomly drop out any of the information
the network has learned. Using model.eval() switches the model to evaluation mode
for inference, disabling the dropout layers of the model. If we plan to continue pre-
training a model later—for example, using the train_model_simple function we
defined earlier in this chapter—saving the optimizer state is also recommended.
Adaptive optimizers such as AdamW store additional parameters for each model
weight. AdamW uses historical data to adjust learning rates for each model parameter
dynamically. Without it, the optimizer resets, and the model may learn suboptimally
or even fail to converge properly, which means it will lose the ability to generate coher-
ent text. Using torch.save, we can save both the model and optimizer state_dict
contents:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },
   "model_and_optimizer.pth"
)
```
Then we can restore the model and optimizer states by first loading the saved data via
torch.load and then using the load_state_dict method:

```python
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()
```

## Loading pretrained weights from OpenAI
Previously, we trained a small GPT-2 model using a limited dataset comprising a short-
story book. This approach allowed us to focus on the fundamentals without the need
for extensive time and computational resources.
Fortunately, OpenAI openly shared the weights of their GPT-2 models, thus elimi-
nating the need to invest tens to hundreds of thousands of dollars in retraining the
model on a large corpus ourselves. So, let’s load these weights into our GPTModel class
and use the model for text generation. Here, weights refer to the weight parameters
stored in the .weight attributes of PyTorch’s Linear and Embedding layers, for exam-
ple. We accessed them earlier via model.parameters() when training the model. In
chapter 6, will reuse these pretrained weights to fine-tune the model for a text classifi-
cation task and follow instructions similar to ChatGPT.
Note that OpenAI originally saved the GPT-2 weights via TensorFlow, which we
have to install to load the weights in Python. The following code will use a progress
bar tool called tqdm to track the download process, which we also have to install.

The download code is relatively long, mostly boilerplate, and not very interesting.
Hence, instead of devoting precious space to discussing Python code for fetching files
from the internet, we download the gpt_download.py Python module directly from
this chapter’s online repository:
```python
import urllib.request
url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)
```
Next, after downloading this file to the local directory of your Python session, you
should briefly inspect the contents of this file to ensure that it was saved correctly and
contains valid Python code.

We can now import the download_and_load_gpt2 function from the gpt_download
.py file as follows, which will load the GPT-2 architecture settings (settings) and
weight parameters (params) into our Python session:

```python
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)
```

Assuming the execution of the previous code has completed, let’s inspect the contents
of settings and params:

print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())

The contents are:

Settings: {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12,
'n_layer': 12}
Parameter dictionary keys: dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])

Both settings and params are Python dictionaries. The settings dictionary stores the
LLM architecture settings similarly to our manually defined GPT_CONFIG_124M settings.
The params dictionary contains the actual weight tensors. Note that we only printed
the dictionary keys because printing the weight contents would take up too much
screen space; however, we can inspect these weight tensors by printing the whole dic-
tionary via print(params) or by selecting individual tensors via the respective dictio-
nary keys, for example, the embedding layer weights:
```python
print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)
```
We downloaded and loaded the weights of the smallest GPT-2 model via the download_
and_load_gpt2(model_size="124M", ...) setting. OpenAI also shares the weights of
larger models: 355M, 774M, and 1558M. The overall architecture of these differently
sized GPT models is the same, as illustrated in figure below, except that different
architectural elements are repeated different numbers of times and the embedding
size differs. The remaining code in this chapter is also compatible with these larger
models.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image101.png?raw=true)

After loading the GPT-2 model weights into Python, we still need to transfer them
from the settings and params dictionaries into our GPTModel instance. First, we cre-
ate a dictionary that lists the differences between the different GPT model sizes in
figure above:

```python
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
```

Suppose we are interested in loading the smallest model, "gpt2-small (124M)". We can
use the corresponding settings from the model_configs table to update our full-length
GPT_CONFIG_124M we defined and used earlier:
```python
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
```
Careful readers may remember that we used a 256-token length earlier, but the origi-
nal GPT-2 models from OpenAI were trained with a 1,024-token length, so we have to
update the NEW_CONFIG accordingly:

NEW_CONFIG.update({"context_length": 1024})

Also, OpenAI used bias vectors in the multi-head attention module’s linear layers to
implement the query, key, and value matrix computations. Bias vectors are not com-
monly used in LLMs anymore as they don’t improve the modeling performance and
are thus unnecessary. However, since we are working with pretrained weights, we need
to match the settings for consistency and enable these bias vectors:

NEW_CONFIG.update({"qkv_bias": True})

We can now use the updated NEW_CONFIG dictionary to initialize a new GPTModel
instance:

```python
gpt = GPTModel(NEW_CONFIG)
gpt.eval()
```
By default, the GPTModel instance is initialized with random weights for pretraining.
The last step to using OpenAI’s model weights is to override these random weights
with the weights we loaded into the params dictionary. For this, we will first define a
small assign utility function that checks whether two tensors or arrays (left and
right) have the same dimensions or shape and returns the right tensor as trainable
PyTorch parameters:

```python
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
            "Right: {right.shape}"
)
    return torch.nn.Parameter(torch.tensor(right))
```
Next, we define a load_weights_into_gpt function that loads the weights from the
params dictionary into a GPTModel instance gpt.(you can find the code in the github for chapter 5 with the function of load_weights_into_gpt)

Developing the load_weights_into_gpt function took a lot of guesswork since
OpenAI used a slightly different naming convention from ours. However, the assign
function would alert us if we try to match two tensors with different dimensions. Also,
if we made a mistake in this function, we would notice this, as the resulting GPT
model would be unable to produce coherent text.

In the load_weights_into_gpt function, we carefully match the weights from
OpenAI’s implementation with our GPTModel implementation. To pick a specific
example, OpenAI stored the weight tensor for the output projection layer for the
first transformer block as params["blocks"][0]["attn"]["c_proj"]["w"]. In our
implementation, this weight tensor corresponds to gpt.trf_blocks[b].att.out_proj
.weight, where gpt is a GPTModel instance.
Developing the load_weights_into_gpt function took a lot of guesswork since
OpenAI used a slightly different naming convention from ours. However, the assign
function would alert us if we try to match two tensors with different dimensions. Also,
if we made a mistake in this function, we would notice this, as the resulting GPT
model would be unable to produce coherent text.
Let’s now try the load_weights_into_gpt out in practice and load the OpenAI
model weights into our GPTModel instance gpt:
```python
load_weights_into_gpt(gpt, params)
gpt.to(device)
```
If the model is loaded correctly, we can now use it to generate new text using our pre-
vious generate function:
```python
torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```

The resulting text is as follows:

Output text:
Every effort moves you toward finding an ideal new way to practice
something!
What makes us want to be on top of that?

We can be confident that we loaded the model weights correctly because the model can
produce coherent text. A tiny mistake in this process would cause the model to fail. In
the following chapters, we will work further with this pretrained model and fine-tune it
to classify text and follow instructions.

# CHAPTER 6

So far, we have coded the LLM architecture, pretrained it, and learned how to
import pretrained weights from an external source, such as OpenAI, into our
model. Now we will reap the fruits of our labor by fine-tuning the LLM on a specific
target task, such as classifying text. The concrete example we examine is classifying
text messages as “spam” or “not spam.” Figure below highlights the two main ways of
fine-tuning an LLM: fine-tuning for classification (step 8) and fine-tuning to follow
instructions (step 9).


![alt text](https://github.com/Rezashatery/LLM/blob/main/image102.png?raw=true)

## Different categories of fine-tuning

The most common ways to fine-tune language models are instruction fine-tuning and
classification fine-tuning. Instruction fine-tuning involves training a language model on
a set of tasks using specific instructions to improve its ability to understand and exe-
cute tasks described in natural language prompts, as illustrated in figure below

![alt text](https://github.com/Rezashatery/LLM/blob/main/image103.png?raw=true)

Two different instruction fine-tuning scenarios. At the top, the model is tasked with determining
whether a given text is spam. At the bottom, the model is given an instruction on how to translate an English
sentence into German.

In classification fine-tuning, a concept you might already be acquainted with if you
have a background in machine learning, the model is trained to recognize a specific 
set of class labels, such as “spam” and “not spam.” Examples of classification tasks extend
beyond LLMs and email filtering: they include identifying different species of plants
from images; categorizing news articles into topics like sports, politics, and technology;
and distinguishing between benign and malignant tumors in medical imaging.
The key point is that a classification fine-tuned model is restricted to predicting
classes it has encountered during its training. For instance, it can determine whether
something is “spam” or “not spam,” as illustrated in figure below, but it can’t say anything
else about the input text.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image104.png?raw=true)

### Choosing the right approach
Instruction fine-tuning improves a model’s ability to understand and generate responses
based on specific user instructions. Instruction fine-tuning is best suited for models
that need to handle a variety of tasks based on complex user instructions, improving
flexibility and interaction quality. Classification fine-tuning is ideal for projects requir-
ing precise categorization of data into predefined classes, such as sentiment analy-
sis or spam detection.
While instruction fine-tuning is more versatile, it demands larger datasets and greater
computational resources to develop models proficient in various tasks. In contrast,
classification fine-tuning requires less data and compute power, but its use is con-
fined to the specific classes on which the model has been trained.



## Preparing the dataset

We will modify and classification fine-tune the GPT model we previously implemented
and pretrained. We begin by downloading and preparing the dataset, as highlighted
in figure below. To provide an intuitive and useful example of classification fine-tuning,
we will work with a text message dataset that consists of spam and non-spam messages.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image105.png?raw=true)

The first step is to download the dataset. SMSSpamCollection.tsv, in the sms_spam_collection folder. 
We can load it into a  pandas DataFrame as follows:

```python
import pandas as pd
df = pd.read_csv(
data_file_path, sep="\t", header=None, names=["Label", "Text"]
)
df
```

![alt text](https://github.com/Rezashatery/LLM/blob/main/image106.png?raw=true)

Let’s examine the class label distribution:
print(df["Label"].value_counts())

Executing the previous code, we find that the data contains “ham” (i.e., not spam) far
more frequently than “spam”:

Label
ham    4825
spam   747

For simplicity, and because we prefer a small dataset (which will facilitate faster fine-
tuning of the LLM), we choose to undersample the dataset to include 747 instances
from each class.
We can use the code in the following listing to undersample and create a balanced
dataset.

```python
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
    num_spam, random_state=123
    )
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
])
    return balanced_df

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())

```
Label
ham    747
spam   747

Next, we convert the “string” class labels "ham" and "spam" into integer class labels 0
and 1, respectively:
```python
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
```
This process is similar to converting text into token IDs. However, instead of using the
GPT vocabulary, which consists of more than 50,000 words, we are dealing with just
two token IDs: 0 and 1.
Next, we create a random_split function to split the dataset into three parts: 70%
for training, 10% for validation, and 20% for testing. (These ratios are common in
machine learning to train, adjust, and evaluate models.)

Let’s save the dataset as CSV (comma-separated value) files so we can reuse it later:
```python
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)
```


## Creating data loaders

We will develop PyTorch data loaders conceptually similar to those we implemented
while working with text data. Previously, we utilized a sliding window technique to
generate uniformly sized text chunks, which we then grouped into batches for more
efficient model training. Each chunk functioned as an individual training instance.
However, we are now working with a spam dataset that contains text messages of vary-
ing lengths. To batch these messages as we did with the text chunks, we have two pri-
mary options:

1) Truncate all messages to the length of the shortest message in the dataset or batch.
2) Pad all messages to the length of the longest message in the dataset or batch.

The first option is computationally cheaper, but it may result in significant informa-
tion loss if shorter messages are much smaller than the average or longest messages,

potentially reducing model performance. So, we opt for the second option, which
preserves the entire content of all messages.
To implement batching, where all messages are padded to the length of the lon-
gest message in the dataset, we add padding tokens to all shorter messages. For this
purpose, we use "<|endoftext|>" as a padding token.
However, instead of appending the string "<|endoftext|>" to each of the text
messages directly, we can add the token ID corresponding to "<|endoftext|>" to the
encoded text messages, as illustrated in figure below. 50256 is the token ID of the padding
token "<|endoftext|>". We can double-check whether the token ID is correct by
encoding the "<|endoftext|>" using the GPT-2 tokenizer from the tiktoken package
that we used previously:

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
```

![alt text](https://github.com/Rezashatery/LLM/blob/main/image107.png?raw=true)

Indeed, executing the preceding code returns [50256].
We first need to implement a PyTorch Dataset, which specifies how the data is
loaded and processed before we can instantiate the data loaders. For this purpose,
we define the SpamDataset class, which implements the concepts in figure above. This
SpamDataset class handles several key tasks: it identifies the longest sequence in the
training dataset, encodes the text messages, and ensures that all other sequences are
padded with a padding token to match the length of the longest sequence.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image108.png?raw=true)

![alt text](https://github.com/Rezashatery/LLM/blob/main/image109.png?raw=true)

The SpamDataset class loads data from the CSV files we created earlier, tokenizes
the text using the GPT-2 tokenizer from tiktoken, and allows us to pad or truncate
the sequences to a uniform length determined by either the longest sequence or a
predefined maximum length. This ensures each input tensor is of the same size,
which is necessary to create the batches in the training data loader we implement
next:

```python
train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)
```
The longest sequence length is stored in the dataset’s max_length attribute. If you are
curious to see the number of tokens in the longest sequence, you can use the follow-
ing code:

```python
print(train_dataset.max_length)
```
The code outputs 120, showing that the longest sequence contains no more than
120 tokens, a common length for text messages. The model can handle sequences
of up to 1,024 tokens, given its context length limit. If your dataset includes longer
texts, you can pass max_length=1024 when creating the training dataset in the pre-
ceding code to ensure that the data does not exceed the model’s supported input
(context) length.
Next, we pad the validation and test sets to match the length of the longest train-
ing sequence. Importantly, any validation and test set samples exceeding the length of
the longest training example are truncated using encoded_text[:self.max_length]
in the SpamDataset code we defined earlier. This truncation is optional; you can set
max_length=None for both validation and test sets, provided there are no sequences
exceeding 1,024 tokens in these sets:

```python
val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
```
Using the datasets as inputs, we can instantiate the data loaders similarly to when we
were working with text data. However, in this case, the targets represent class labels
rather than the next tokens in the text. For instance, if we choose a batch size of 8,
each batch will consist of eight training examples of length 120 and the correspond-
ing class label of each example, as illustrated in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image110.png?raw=true)


```python
from torch.utils.data import DataLoader
num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

```
Lastly, to get an idea of the dataset size, let’s print the total number of batches in
each dataset:

```python
print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")
```
The number of batches in each dataset are
130 training batches
19 validation batches
38 test batches




## Initializing a model with pretrained weights
We must prepare the model for classification fine-tuning to identify spam messages.
We start by initializing our pretrained model, as highlighted in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image111.png?raw=true)

To begin the model preparation process, we employ the same configurations we used
to pretrain unlabeled data:

```python
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

```

Next, we import the download_and_load_gpt2 function from the gpt_download.py
file and reuse the GPTModel class and load_weights_into_gpt function from pretrain-
ing to load the downloaded weights into the GPT model.

```python
from gpt_download import download_and_load_gpt2
from chapter05 import GPTModel, load_weights_into_gpt
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
)
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
```
After loading the model weights into the GPTModel, we reuse the text generation utility 
function to ensure that the model generates coherent text:

```python
from chapter04 import generate_text_simple
from chapter05 import text_to_token_ids, token_ids_to_text
text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))
```
The following output shows the model generates coherent text, which is indicates that
the model weights have been loaded correctly:

Every effort moves you forward.
The first step is to understand the importance of your work

Before we start fine-tuning the model as a spam classifier, let’s see whether the model
already classifies spam messages by prompting it with instructions:

```python
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))
```

The model output is:

Is the following text 'spam'? Answer with 'yes' or 'no': 'You are a winner
you have been specially selected to receive $1000 cash
or a $2000 award.'
The following text 'spam'? Answer with 'yes' or 'no': 'You are a winner

Based on the output, it’s apparent that the model is struggling to follow instructions.
This result is expected, as it has only undergone pretraining and lacks instruction
fine-tuning. So, let’s prepare the model for classification fine-tuning.


## Adding a classification head

We must modify the pretrained LLM to prepare it for classification fine-tuning. To do
so, we replace the original output layer, which maps the hidden representation to a
vocabulary of 50,257, with a smaller output layer that maps to two classes: 0 (“not
spam”) and 1 (“spam”), as shown in figure below. We use the same model as before, except
we replace the output layer.

### Output layer nodes
We could technically use a single output node since we are dealing with a binary clas-
sification task. However, it would require modifying the loss function, as I discuss in
“Losses Learned—Optimizing Negative Log-Likelihood and Cross-Entropy in PyTorch”.
Therefore, we choose a more general approach, where the
number of output nodes matches the number of classes. For example, for a three-
class problem, such as classifying news articles as “Technology,” “Sports,” or “Pol-
itics,” we would use three output nodes, and so forth.


![alt text](https://github.com/Rezashatery/LLM/blob/main/image112.png?raw=true)

This output neatly lays out the architecture we laid out in chapter 4. As previously dis-
cussed, the GPTModel consists of embedding layers followed by 12 identical transformer
blocks (only the last block is shown for brevity), followed by a final LayerNorm and the
output layer, out_head.
Next, we replace the out_head with a new output layer that we will fine-tune.

### Fine-tuning selected layers vs. all layers
Since we start with a pretrained model, it’s not necessary to fine-tune all model layers.
In neural network-based language models, the lower layers generally capture basic lan-
guage structures and semantics applicable across a wide range of tasks and datasets.
So, fine-tuning only the last layers (i.e., layers near the output), which are more specific
to nuanced linguistic patterns and task-specific features, is often sufficient to adapt the
model to new tasks. A nice side effect is that it is computationally more efficient to fine-
tune only a small number of layers.

To get the model ready for classification fine-tuning, we first freeze the model, meaning
that we make all layers nontrainable:

```python
for param in model.parameters():
param.requires_grad = False
```
Then, we replace the output layer (model.out_head), which originally maps the layer
inputs to 50,257 dimensions, the size of the vocabulary (see figure above).

```python
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    n_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)
```

To keep the code more general, we use BASE_CONFIG["emb_dim"], which is equal to
768 in the "gpt2-small (124M)" model. Thus, we can also use the same code to work
with the larger GPT-2 model variants.
This new model.out_head output layer has its requires_grad attribute set to
True by default, which means that it’s the only layer in the model that will be
updated during training. Technically, training the output layer we just added is suffi-
cient. However, as I found in experiments, fine-tuning additional layers can notice-
ably improve the predictive performance of the model. We also configure the last transformer block and the final LayerNorm
module, which connects this block to the output layer, to be trainable, as depicted
in figure below.
To make the final LayerNorm and last transformer block trainable, we set their
respective requires_grad to True:

```python
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
```
Even though we added a new output layer and marked certain layers as trainable or
nontrainable, we can still use this model similarly to how we have previously.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image113.png?raw=true)

For instance, we can feed it an example text identical to our previously used example
text:

```python
inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape)
```
The print output shows that the preceding code encodes the inputs into a tensor con-
sisting of four input tokens:

Inputs: tensor([[5211, 345, 423, 640]])
Inputs dimensions: torch.Size([1, 4])

Then, we can pass the encoded token IDs to the model as usual:
```python
with torch.no_grad():
    outputs = model(inputs)
print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape)
```

A similar input would have previously produced an output tensor of [1, 4, 50257],
where 50257 represents the vocabulary size. The number of output rows corresponds
to the number of input tokens (in this case, four). However, each output’s embedding
dimension (the number of columns) is now 2 instead of 50,257 since we replaced the
output layer of the model.
Remember that we are interested in fine-tuning this model to return a class label
indicating whether a model input is “spam” or “not spam.” We don’t need to fine-
tune all four output rows; instead, we can focus on a single output token. In particu-
lar, we will focus on the last row corresponding to the last output token, as shown in
figure below.
To extract the last output token from the output tensor, we use the following code:
```python
print("Last output token:", outputs[:, -1, :])
```
This prints:

Last output token: tensor([[-3.5983, 3.9902]])

We still need to convert the values into a class-label prediction. But first, let’s under-
stand why we are particularly interested in the last output token only.
We have already explored the attention mechanism, which establishes a relationship
between each input token and every other input token, and the concept of a causal
attention mask, commonly used in GPT-like models . This mask restricts a

![alt text](https://github.com/Rezashatery/LLM/blob/main/image114.png?raw=true)

token’s focus to its current position and the those before it, ensuring that each token
can only be influenced by itself and the preceding tokens, as illustrated in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image115.png?raw=true)

The causal attention mechanism, where the attention scores between input tokens are displayed in a
matrix format. The empty cells indicate masked positions due to the causal attention
mask, preventing tokens from attending to future tokens. The values in the cells
represent attention scores; the last token, time, is the only one that computes
attention scores for all preceding tokens.

Given the causal attention mask setup in figure above, the last token in a sequence accu-
mulates the most information since it is the only token with access to data from all the
previous tokens. Therefore, in our spam classification task, we focus on this last token
during the fine-tuning process.
We are now ready to transform the last token into class label predictions and calcu-
late the model’s initial prediction accuracy. Subsequently, we will fine-tune the model
for the spam classification task.



## Calculating the classification loss and accuracy
Only one small task remains before we fine-tune the model: we must implement the
model evaluation functions used during fine-tuning.
Before implementing the evaluation utilities, let’s briefly discuss how we convert
the model outputs into class label predictions. We previously computed the token ID
of the next token generated by the LLM by converting the 50,257 outputs into proba-
bilities via the softmax function and then returning the position of the highest proba-
bility via the argmax function. We take the same approach here to calculate whether
the model outputs a “spam” or “not spam” prediction for a given input, as shown in
figure below. The only difference is that we work with 2-dimensional instead of 50,257-
dimensional outputs.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image116.png?raw=true)

We can obtain the class label:
```python
probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
print("Class label:", label.item())
```

In this case, the code returns 1, meaning the model predicts that the input text is
“spam.” Using the softmax function here is optional because the largest outputs
directly correspond to the highest probability scores. Hence, we can simplify the code
without using softmax:
```python
logits = outputs[:, -1, :]
label = torch.argmax(logits)
print("Class label:", label.item())
```
To determine the classification accuracy, we apply the argmax-based prediction
code to all examples in the dataset and calculate the proportion of correct predictions
by defining a calc_accuracy_loader function.

```python
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
            logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (
            (predicted_labels == target_batch).sum().item())
    else:
         break
    return correct_predictions / num_examples
```
Let’s use the function to determine the classification accuracies across various datasets
estimated from 10 batches for efficiency:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(
    train_loader, model, device, num_batches=10
)
val_accuracy = calc_accuracy_loader(
    val_loader, model, device, num_batches=10
)
test_accuracy = calc_accuracy_loader(
    test_loader, model, device, num_batches=10
)
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")
```

Via the device setting, the model automatically runs on a GPU if a GPU with Nvidia
CUDA support is available and otherwise runs on a CPU. The output is:

Training accuracy: 46.25%
Validation accuracy: 45.00%
Test accuracy: 48.75%

As we can see, the prediction accuracies are near a random prediction, which would be
50% in this case. To improve the prediction accuracies, we need to fine-tune the model.
However, before we begin fine-tuning the model, we must define the loss function
we will optimize during training. Our objective is to maximize the spam classification
accuracy of the model, which means that the preceding code should output the cor-
rect class labels: 0 for non-spam and 1 for spam.
Because classification accuracy is not a differentiable function, we use cross-
entropy loss as a proxy to maximize accuracy. Accordingly, the calc_loss_batch func-
tion remains the same, with one adjustment: we focus on optimizing only the last
token, model(input_batch)[:, -1, :], rather than all tokens, model(input_batch):

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss
```
We use the calc_loss_batch function to compute the loss for a single batch obtained
from the previously defined data loaders. To calculate the loss for all batches in a data
loader, we define the calc_loss_loader function as before.

```python
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)

    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
             break
    return total_loss / num_batches
```
Similar to calculating the training accuracy, we now compute the initial loss for each
data set:

```python
with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")
```
The initial loss values are: 

Training loss: 2.453
Validation loss: 2.583
Test loss: 2.322

Next, we will implement a training function to fine-tune the model, which means
adjusting the model to minimize the training set loss. Minimizing the training set loss
will help increase the classification accuracy, which is our overall goal.




## Fine-tuning the model on supervised data
We must define and use the training function to fine-tune the pretrained LLM and
improve its spam classification accuracy. The training loop, illustrated in figure below,
is the same overall training loop we used for pretraining; the only difference is that
we calculate the classification accuracy instead of generating a sample text to evalu-
ate the model.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image117.png?raw=true)

The training function implementing the concepts shown in figure above also closely mir-
rors the train_model_simple function used for pretraining the model. The only two dis-
tinctions are that we now track the number of training examples seen (examples_seen)
instead of the number of tokens, and we calculate the accuracy after each epoch instead
of printing a sample text.

Figure below plots the resulting loss curves.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image118.png?raw=true)

As we can see based on the sharp downward slope in figure above, the model is learning
well from the training data, and there is little to no indication of overfitting; that is,
there is no noticeable gap between the training and validation set losses.

### Choosing the number of epochs
Earlier, when we initiated the training, we set the number of epochs to five. The num-
ber of epochs depends on the dataset and the task’s difficulty, and there is no uni-
versal solution or recommendation, although an epoch number of five is usually a
good starting point. If the model overfits after the first few epochs as a loss plot (see
figure above), you may need to reduce the number of epochs. Conversely, if the trend-
line suggests that the validation loss could improve with further training, you should
increase the number of epochs. In this concrete case, five epochs is a reasonable
number as there are no signs of early overfitting, and the validation loss is close to 0.

when using the train_classifier_simple function, which means our estimations of
training and validation performance are based on only five batches for efficiency
during training.
Now we must calculate the performance metrics for the training, validation, and
test sets across the entire dataset by running the following code, this time without
defining the eval_iter value:

```python
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")
```
The resulting accuracy values are:

Training accuracy: 97.21%
Validation accuracy: 97.32%
Test accuracy: 95.67%

The training and test set performances are almost identical. The slight discrepancy
between the training and test set accuracies suggests minimal overfitting of the train-
ing data. Typically, the validation set accuracy is somewhat higher than the test set
accuracy because the model development often involves tuning hyperparameters to
perform well on the validation set, which might not generalize as effectively to the test
set. This situation is common, but the gap could potentially be minimized by adjusting
the model’s settings, such as increasing the dropout rate (drop_rate) or the weight_
decay parameter in the optimizer configuration.

## Using the LLM as a spam classifier
Having fine-tuned and evaluated the model, we are now ready to classify spam mes-
sages (see figure below). Let’s use our fine-tuned GPT-based spam classification model.
The following classify_review function follows data preprocessing steps similar
to those we used in the SpamDataset implemented earlier. Then, after processing
text into token IDs, the function uses the model to predict an integer class label,
similar to what we implemented in section before, and then returns the corresponding
class name.
![alt text](https://github.com/Rezashatery/LLM/blob/main/image119.png?raw=true)

# Fine-tuning to follow instructions (CHAPTER 7)

Previously, we implemented the LLM architecture, carried out pretraining, and
imported pretrained weights from external sources into our model. Then, we
focused on fine-tuning our LLM for a specific classification task: distinguishing
between spam and non-spam text messages. Now we’ll implement the process for
fine-tuning an LLM to follow human instructions, as illustrated in figure below.
Instruction fine-tuning is one of the main techniques behind developing LLMs for
chatbot applications, personal assistants, and other conversational tasks.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image120.png?raw=true)

## Introduction to instruction fine-tuning
We now know that pretraining an LLM involves a training procedure where it learns
to generate one word at a time. The resulting pretrained LLM is capable of text comple-
tion, meaning it can finish sentences or write text paragraphs given a fragment as
input. However, pretrained LLMs often struggle with specific instructions, such as “Fix
the grammar in this text” or “Convert this text into passive voice.” Later, we will exam-
ine a concrete example where we load the pretrained LLM as the basis for instruction
fine-tuning, also known as supervised instruction fine-tuning.
Here, we focus on improving the LLM’s ability to follow such instructions and gen-
erate a desired response, as illustrated in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image121.png?raw=true)

Preparing the dataset is a key aspect of instruction fine-tuning. Then we’ll complete
all the steps in the three stages of the instruction fine-tuning process, beginning 
with the dataset preparation, as shown in figure below.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image122.png?raw=true)

## Preparing a dataset for supervised instruction fine-tuning

Let’s download and format the instruction dataset for instruction fine-tuning a pre-
trained LLM. The dataset consists of 1,100 instruction–response pairs similar to those in
figure above. This dataset was created specifically for this book.
The following code implements and executes a function to download this dataset,
which is a relatively small file (only 204 KB) in JSON format. JSON, or JavaScript Object
Notation, mirrors the structure of Python dictionaries, providing a simple structure
for data interchange that is both human readable and machine friendly.

```python
import json
import os
import urllib
def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    with open(file_path, "r") as file:
        data = json.load(file)
    return data
file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)
data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))

```
The output of executing the preceding code is:

Number of entries: 1100

The data list that we loaded from the JSON file contains the 1,100 entries of the
instruction dataset. Let’s print one of the entries to see how each entry is structured:

The content of the example entry is
Example entry:
```python
{'instruction': 'Identify the correct spelling of the following word.',
'input': 'Ocassion', 'output': "The correct spelling is 'Occasion.'"}
```
As we can see, the example entries are Python dictionary objects containing an
'instruction', 'input', and 'output'.

Instruction fine-tuning involves training a model on a dataset where the input-output
pairs, like those we extracted from the JSON file, are explicitly provided. There are
various methods to format these entries for LLMs. Figure below illustrates two different

![alt text](https://github.com/Rezashatery/LLM/blob/main/image123.png?raw=true)

example formats, often referred to as prompt styles, used in the training of notable
LLMs such as Alpaca and Phi-3. 
The rest of this chapter uses the Alpaca prompt style since it is one of the most
popular ones, largely because it helped define the original approach to fine-tuning.

Let’s define a format_input function that we can use to convert the entries in the
data list into the Alpaca-style input format.

```python
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text
```

Note that the format_input skips the optional ### Input: section if the 'input' field
is empty, which we can test out by applying the format_input function to entry
data[999] that we inspected earlier:
```python
model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
print(model_input + desired_response)
```
The output shows that entries with an empty 'input' field don’t contain an ###
Input: section in the formatted input:
```python
Below is an instruction that describes a task. Write a response that
appropriately completes the request.
### Instruction:
What is an antonym of 'complicated'?
### Response:
An antonym of 'complicated' is 'simple'.
```
Before we move on to setting up the PyTorch data loaders in the next section, let’s
divide the dataset into training, validation, and test sets analogous to what we have
done with the spam classification dataset in the previous chapter. The following listing
shows how we calculate the portions.

```python
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))
```
This partitioning results in the following dataset sizes:

Training set length: 935
Validation set length: 55
Test set length: 110
Next, we focus on developing the method for constructing the training batches for fine-tuning the LLM.

## Organizing data into training batches
As we progress into the implementation phase of our instruction fine-tuning process,
the next step, illustrated in figure below, focuses on constructing the training batches
effectively. This involves defining a method that will ensure our model receives the
formatted training data during the fine-tuning process.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image124.png?raw=true)

In the previous chapter, the training batches were created automatically by the PyTorch
DataLoader class, which employs a default collate function to combine lists of samples
into batches. A collate function is responsible for taking a list of individual data sam-
ples and merging them into a single batch that can be processed efficiently by the
model during training.

However, the batching process for instruction fine-tuning is a bit more involved
and requires us to create our own custom collate function that we will later plug into
the DataLoader. We implement this custom collate function to handle the specific
requirements and formatting of our instruction fine-tuning dataset.
Let’s tackle the batching process in several steps, including coding the custom col-
late function, as illustrated in figure below First, to implement steps 2.1 and 2.2, we
code an InstructionDataset class that applies format_input and pretokenizes all
inputs in the dataset, similar to the SpamDataset in chapter 6.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image125.png?raw=true)

This two-step process,
detailed in figure below, is implemented in the __init__ constructor method of the
InstructionDataset.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image126.png?raw=true)

```python
import torch
from torch.utils.data import Dataset
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
            tokenizer.encode(full_text)
        )
    def __getitem__(self, index):
        return self.encoded_texts[index]
    def __len__(self):
        return len(self.data)   
```
Similar to the approach used for classification fine-tuning, we want to accelerate train-
ing by collecting multiple training examples in a batch, which necessitates padding all
inputs to a similar length. As with classification fine-tuning, we use the <|endoftext|>
token as a padding token.

Instead of appending the <|endoftext|> tokens to the text inputs, we can append
the token ID corresponding to <|endoftext|> to the pretokenized inputs directly. We
can use the tokenizer’s .encode method on an <|endoftext|> token to remind us
which token ID we should use:
```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
```
The resulting token ID is 50256.

Moving on to step 2.3 of the process, we adopt a more sophisti-
cated approach by developing a custom collate function that we can pass to the data
loader. This custom collate function pads the training examples in each batch to the
same length while allowing different batches to have different lengths, as demon-
strated in figure below. This approach minimizes unnecessary padding by only extending
sequences to match the longest one in each batch, not the whole dataset.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image127.png?raw=true)

We can implement the padding process with a custom collate function:

```python
def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cpu"
    ):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst = []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
          new_item + [pad_token_id] *
          (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor
```
The custom_collate_draft_1 we implemented is designed to be integrated into a
PyTorch DataLoader, but it can also function as a standalone tool. Here, we use it
independently to test and verify that it operates as intended.

We have just implemented our first custom collate function to create batches from
lists of inputs. However, as we previously learned, we also need to create batches with
the target token IDs corresponding to the batch of input IDs. These target IDs, as
shown in figure below, are crucial because they represent what we want the model to
generate and what we need during training to calculate the loss for the weight
updates. That is, we modify our custom collate function to return the target token IDs
in addition to the input token IDs.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image128.png?raw=true)

Similar to the process we used to pretrain an LLM, the target token IDs match the
input token IDs but are shifted one position to the right. This setup, as shown in fig-
ure below, allows the LLM to learn how to predict the next token in a sequence.

![alt text](https://github.com/Rezashatery/LLM/blob/main/image129.png?raw=true)

The following updated collate function generates the target token IDs from the input
token IDs: