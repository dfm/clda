This is research code and it's probably still broken. That being said, feel
free to take a look around and use whatever you want under the terms of the
MIT license (see LICENSE).

Usage
-----

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
