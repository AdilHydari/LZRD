# Range Coder Test Data

This repository contains tools to generate test data for range coder compression. Range coders work best with data that has non-uniform probability distributions, such as text.

## Tools Included

1. **text_generator.py** - Generate synthetic test data with different distribution patterns
2. **train_markov.py** - Create more realistic text by training a Markov model on existing text

## Quick Start

### Generate basic test data

```bash
# Generate 10KB of Markov text
python text_generator.py --output test_data.bin --size 10000 --type markov

# Generate text with skewed character distribution
python text_generator.py --output skewed_data.bin --size 10000 --type skewed

# Generate text with repeated patterns
python text_generator.py --output pattern_data.bin --size 10000 --type patterns
```

### Generate Markov text from a real source

1. First, find a suitable text file to use as training data. Project Gutenberg is a good source:
   - Download a book: https://www.gutenberg.org/browse/scores/top
   - Example: download "Pride and Prejudice" as a text file

2. Train a Markov model and generate sample text:

```bash
# Train on book.txt and generate 20KB of text
python train_markov.py --input book.txt --output markov_book.bin --size 20000 --order 4
```

## Using with the Range Coder

Once you've generated test data, you can use it with the range coder:

```bash
# Assuming your range coder is compiled as 'rangecoder'
./rangecoder test_data.bin compressed.bin decompressed.bin
```

## Tips for Good Test Data

1. **Text data** is excellent for range coders due to its non-uniform character distribution
2. **Higher-order Markov models** (--order 4 or 5) produce more realistic text patterns
3. **Source code files** are also good test cases as they have highly repetitive patterns
4. **HTML/XML** files typically compress well due to repetitive tags
5. **Generated pattern data** can be useful for testing extreme cases

## Where to Find More Test Data

- Project Gutenberg: https://www.gutenberg.org/
- Canterbury Corpus: http://corpus.canterbury.ac.nz/
- Large Text Compression Benchmark: http://mattmahoney.net/dc/text.html 