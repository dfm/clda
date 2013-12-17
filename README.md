This is research code and it's probably still broken. That being said, feel
free to take a look around and use whatever you want under the terms of the
MIT license (see LICENSE).

Usage
-----

**Building a topic model on ArXiv metadata** —
The first step is to download the pre-parsed dataset from
[data.arxiv.io/abstracts.db.gz](http://data.arxiv.io/abstracts.db.gz). It
is saved a SQLite database with a single table (called `articles`) and the
titles and abstracts have been tokenized using the nltk PennTreebank
tokenizer. Once you've downloaded this database, you can build a topic model
with 200 topics by running
```
scripts/run-lda results -k 200
```
This script with save files in the `results` directory. In particular, it
will save `model.*.pkl` files that are snapshots of the LDA model object
as a function of time. To look at the top-N word distributions for a
particular model (number 390, for example; look in the results directory
for the names of the snapshots that are saved), run
```
scripts/lda-results results/model.0390.pkl
```
Similarly, to infer the topic distributions for a particualr document ([my
last paper](http://arxiv.org/abs/1202.3665), for example), run
```
scripts/lda-infer results/model.0390.pkl "1202.3665"
```

**Preparing the collaborative filtering dataset** —
First, you need to request access to the CiteULike dataset by following [the
instructions](http://www.citeulike.org/faq/data.adp) and then download the
files called [linkouts](http://static.citeulike.org/data/linkouts.bz2) and
[current](http://static.citeulike.org/data/current.bz2). Unzip these files
into the same directory. Then, download the SQLite database of ArXiv metadata
provided at [data.arxiv.io](http://data.arxiv.io/abstracts.db.gz) and unzip
it. Then run the parsing script

```
scripts/prepare-citeulike-data /path/to/citulike/data/ /path/to/abstracts.db
```

to create a new table (called `citeulike`) in SQLite database giving the
parsed CiteULike listings.
