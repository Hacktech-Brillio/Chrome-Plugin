// preprocessing.js

// You need a tokenizer, stop word removal, lemmatizer, and a mapping from words to indices.

let stopWords = [];
let wordToIdx = {};

// Load stop words and word_to_idx dictionary
fetch(chrome.runtime.getURL('stop_words.json'))
  .then(response => response.json())
  .then(data => {
    stopWords = new Set(data);
  });

fetch(chrome.runtime.getURL('word_to_idx.json'))
  .then(response => response.json())
  .then(data => {
    wordToIdx = data;
  });

// Preprocessing function
async function preprocessText(text) {
  // Expand contractions (implement a contractions dictionary if needed)
  text = text.toLowerCase();
  text = text.replace(/[^a-z0-9\s]/g, ''); // Remove special characters

  // Tokenize
  let tokens = text.split(/\s+/); // Basic whitespace tokenizer

  // Remove stop words
  tokens = tokens.filter(token => !stopWords.has(token));

  // Lemmatize tokens (you may need to use a library for this or keep it simple)
  // For this example, we are skipping lemmatization.

  // Convert tokens to indices
  let indices = tokens.map(token => wordToIdx[token] || wordToIdx['UNK']);

  // Pad sequences
  const maxSeqLength = 100;
  if (indices.length < maxSeqLength) {
    indices = indices.concat(new Array(maxSeqLength - indices.length).fill(wordToIdx['PAD']));
  } else {
    indices = indices.slice(0, maxSeqLength);
  }

  // Create an Int64Array and return it as an ONNX tensor
  const inputTensor = new ort.Tensor('int64', new Int32Array(indices), [1, maxSeqLength]);
  return inputTensor;
}
