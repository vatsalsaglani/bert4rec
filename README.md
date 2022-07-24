# Encoder based Self-Attention model for Sequential Recommendation

This repository contains a Self-Attention model for sequential item recommendation. The model is trained using the [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/).

The architecture of this model is based on the _next word prediction_ task in NLP. There is a small twist, we can predict the item which can come before the given set of items as well.


## How to train the model?

Given the cronological sequence of movies watched by the user, we randomly replace certain number of movies with our `MASK` i.e. `1` in the input sequence. While training, our model learns to predict the values which are masked.

For example we have the following sequence,

```
# start seq
[3, 4, 99, 3330, 334, 567]
```

We randomly mask two items in the sequence and make our input sequence as
```
# input seq
[1, 4, 99, 1, 334, 567]
```

Now the `input seq` becomes our start sequence and the `start seq` i.e. the original sequence becomes our target sequence. 