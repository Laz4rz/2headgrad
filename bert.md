## Bert

Bert is an encoder based model. Which means that when you see the big-ass transformer diagram, like the one below. Bert is the left part. 

![alt text](images/bert_transformer.png)

As you can, it doesnt have a projection to vocabulary size at the end, nor the softmax producing probabilities for next token. So what does it do?

![alt text](images/bert.png)

As we established in `attention.ipynb`, the difference between encoder and decoder blocks is that the latter can't see all of the tokens, it can see the future, as it's tasks is to PREDICT the next token, based on the ones preceeding it. 

Encoder on the other hand is meant to see the whole input, and give the best information possible about the whole thing, and how all tokens interact with each other. This is why encoder are used for embeddings. They're currently our best chance at representing inputs in some high dimensional space. So that we can build RAGs, cluster by similarity, or whatever we'd like to do knowing what inputs are similar to each other. 

### Word embeddings

