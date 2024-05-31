# Human Name Generator with Transformer Decoder

This project implements a model to generate human names using a Transformer-based decoder architecture. The model is trained on a dataset of names and can generate new names by predicting the next character in a sequence.

## Table of Contents
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

## Introduction

The goal of this project is to create a generative model that can produce realistic human names. The model leverages a Transformer-based architecture, which is known for its effectiveness in sequence-to-sequence tasks such as language modeling and text generation.

## Model Architecture

The model architecture includes several key components:

1. **TextDataset**: A custom dataset class to handle the preprocessing and batching of the input data.
2. **DecoderHead**: A single head of the multi-head attention mechanism.
3. **MultiHead**: Combines multiple `DecoderHead` instances to implement multi-head attention.
4. **TextGenerator**: The main model class that integrates token embedding, positional embedding, multi-head attention, and a feedforward neural network.

### TextDataset

The `TextDataset` class prepares the data for training and validation. It generates input (`x`) and target (`y`) tensors for the model, which are sequences of indices representing characters.

### DecoderHead

The `DecoderHead` class implements a single attention head, including linear transformations for keys, queries, and values, as well as the attention mechanism with a lower triangular mask to ensure causality.

### MultiHead

The `MultiHead` class combines multiple `DecoderHead` instances to perform multi-head attention, which allows the model to focus on different parts of the input sequence simultaneously.

### TextGenerator

The `TextGenerator` class is the main model, combining token embeddings, positional embeddings, the multi-head attention mechanism, a feedforward neural network, and a final linear layer to generate output logits.

## Dataset

The dataset used for training the model consists of a list of human names stored in a text file (`names.txt`). Each name is preprocessed and split into sequences of characters, which are then converted into indices based on the vocabulary.

## Training

The training script initializes the model, defines the loss function (cross-entropy loss), and sets up the optimizer (AdamW). The model is trained for a specified number of epochs, with the training and validation losses printed after each epoch.

## Usage

1. Prepare a text file `names.txt` with a list of names, each on a new line.
2. Run the training script to train the model.
3. After training, the model will output a sequence of generated names.

## Dependencies

- Python 3.7+
- PyTorch
- NumPy

Install the required packages using:

```sh
pip install -r requirements.txt
```

## Acknowledgments

This project is inspired by the advancements in Transformer-based models for natural language processing. The implementation is a simplified version tailored for the specific task of name generation.