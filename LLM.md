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



# Working with text data
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