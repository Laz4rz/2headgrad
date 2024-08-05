## Bert

Bert is an encoder based model. Which means that when you see the big-ass transformer diagram, like the one below. Bert is the left part. 

![alt text](images/bert_transformer.png)

As you can, it doesnt have a projection to vocabulary size at the end, nor the softmax producing probabilities for next token. So what does it do?

![alt text](images/bert.png)

As we established in `attention.ipynb`, the difference between encoder and decoder blocks is that the latter can't see all of the tokens, it can see the future, as it's tasks is to PREDICT the next token, based on the ones preceeding it. 

Encoder on the other hand is meant to see the whole input, and give the best information possible about the whole thing, and how all tokens interact with each other. This is why encoder are used for embeddings. They're currently our best chance at representing inputs in some high dimensional space. So that we can build RAGs, cluster by similarity, or whatever we'd like to do knowing what inputs are similar to each other. 

Reminder: encoder vs decoder is distinguished intuitively by the above, but in practice, it's just that decoder has upper triangular part of attention matrix masked with zeros (-inf pre-softmax that yields zeros).

### Word embeddings

Starting with the simplest thing. You'd want to embedd a word, using the information from the whole input? How to do it? You put the input into Bert, and as each of the input tokens goes through more and more encoder blocks stacked on each other.

![alt text](images/bert_pass.png)

This image is pretty busy. What I want us to see is that each token passes through stacked on each other encoder blocks, and comes out of each having some more information about all other tokens embedded. Then on the right side the red gradient shows the "amount" of shared information that fills the tokens after each block. 

Now the word embedding is the simplest thing. Take the last vector corresponding to token you wanted embedded. Well, usually you take like the last one, and outputs of few previous encoders block to take a mean over them. This empirically yields better results. 

### Sentence embeddings — CLS Token

Inputs passed to Bert start with the CLS token. This token is used to capture the embedded meaning of the whole input, and in the Bert pretraining phase was used to drive the classification task. 

Therefore it is easy to see, that this token will contain a ton of information about the input. Why? 1. It has no "self-meaning" like other tokens do, 2. It had to perform well for the training classification task. 

In practice, when we want sentence embeddings we again don't take the last output for CLS, but either second to last (as it is less skewed to the classification task used in trainign), or mean of all outputs for CLS token. 

### Sentence embeddings — SentenceBERT

Involves an additional training step for Bert. The weights of the base model are cloned, and two sentences are passed through them. We calculate the similarity of the embedding vectors for both sentences (usually cosine similarity), and put this in the loss function, together with target similarity. Then the gradient is propagated with regard to both inputs. Great source: https://www.youtube.com/watch?v=lVqwznaVi78

We can again use different ways of capturing the embedding for sentence before the similarity calculation: CLS token, max over time for all otputs, mean of all outputs, whatever works. 

