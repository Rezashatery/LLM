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
