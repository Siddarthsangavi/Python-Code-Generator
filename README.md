## So this was the project made by me and my team for a hackathon called Gen-Ai-Rush-Buildathon which was a national hackathon. I would proudly say that our tam was in top 1%. 
## team Members:
@churanta https://github.com/Churanta/
@chandana https://github.com/Chandana-cs
# Gen-Ai-Rush-Buildathon

## A Brief of the Prototype:
The provided code is a prototype for fine-tuning the T5 (Text-to-Text Transfer Transformer) model for a specific task: code generation from natural language queries. The prototype is implemented using TensorFlow and the Hugging Face transformers library, which makes it easier to work with state-of-the-art NLP models.

Here is a brief overview of the prototype's functionality:

1. Import Libraries: The prototype begins by importing the necessary libraries, including TensorFlow, the Hugging Face transformers library, and other utilities.
2. Setup Strategy: The prototype sets up the strategy for distributed training if multiple GPUs are available. It also enables XLA (Accelerated Linear Algebra) to optimize computations on CPUs and mixed-precision training (fp16) for better performance.
3. Download and Preprocess Dataset: The code includes functions to download a dataset from a remote URL and preprocess it. The dataset used is the "mbpp" dataset, containing text-code pairs.
4. Tokenization and Feature Conversion: The prototype defines a function to convert text-code pairs into model-friendly inputs by encoding the text and code sequences using the T5 tokenizer. The inputs are then converted into features for training and evaluation.
5. Model Creation: The T5 model is loaded from a pre-trained checkpoint using the Hugging Face transformers library. The model architecture is TFT5ForConditionalGeneration, a version of T5 for conditional text generation tasks.
6. Training Loop: The prototype defines a custom Trainer class responsible for training the T5 model. The training loop involves iterating over the training dataset and performing forward and backward passes to optimize the model's parameters. The training is distributed across available devices if multiple GPUs are used.
7. Evaluation: The prototype also includes an evaluation loop, where the trained model is evaluated on a validation dataset. The evaluation loop calculates the loss and other metrics to assess the model's performance.
8. Save Model and Tokenizer: After training, the fine-tuned model and tokenizer are saved to the specified output directory for later use.
9. Prediction: The prototype provides two functions for making predictions using the trained model. One function can predict code from randomly selected samples in the dataset, while the other function can predict code from custom user-provided text queries.
  
## Tech Stack: 

The tech stack used in the provided code prototype includes the following components and libraries:
1. Programming Language: Python - The entire prototype is written in Python, a widely-used and versatile programming language.
2. Deep Learning Framework: TensorFlow - TensorFlow is an open-source deep learning framework developed by Google. It is used for building and training neural networks, including Transformer-based models like T5.
3. Hugging Face Transformers Library: The Hugging Face `transformers` library is a popular open-source library that provides pre-trained models, tokenization utilities, and tools for working with transformer-based models in NLP. It simplifies the process of using state-of-the-art NLP models like T5.
4. Transformers Models: The prototype specifically uses the `TFT5ForConditionalGeneration` model, which is a TensorFlow version of the T5 model designed for conditional text generation tasks.
5. Dataset Handling: The prototype uses the `datasets` library, part of the Hugging Face ecosystem, for downloading, loading, and handling datasets. It allows easy access to various datasets for NLP tasks.
6. Data Preprocessing: The prototype uses tokenization and feature conversion functions from the Hugging Face `transformers` library to preprocess the text-code pairs into model-friendly inputs.
7. Distributed Training: TensorFlow's distributed training capability is used to set up a distribution strategy for training across multiple GPUs.
8. Logging and Progress Monitoring: The prototype includes custom logging functions and a progress bar utility to monitor the training and evaluation progress.

The primary focus of the tech stack is to enable the fine-tuning of the T5 model for code generation while providing the necessary tools and utilities for data preprocessing, model handling, and training. The Hugging Face ecosystem plays a crucial role in simplifying many NLP-related tasks, allowing developers to work with advanced NLP models more efficiently.
   
## Step-by-Step Code Execution Instructions:
1. Clone the repository ` https://github.com/Siddarthsangavi/Python-Code-Generator`
2. run the ` training.py ` file
3. wait till the model get trained
4. run ` main.py `
  
## What I Learned:
  The biggest learning we had while developing this prototype was the practical implementation of fine-tuning a transformer-based model for a specific NLP task. I gained insights into dataset preparation, tokenization, training loop, and evaluation processes. Utilizing the Hugging Face `transformers` library simplified complex tasks, such as downloading datasets and managing checkpoints. Additionally, we learned the significance of distributed training in accelerating training on multiple GPUs. Understanding code generation using transformer models highlighted their potential in automating code-writing tasks. Overall, this experience deepened my knowledge of machine learning, NLP, and the capabilities of open-source libraries.

## Architecture
![Picture1](https://github.com/Churanta/Binary-Brains/assets/83538805/58ae9a02-057b-4def-bf84-aade6a35e131)

## Screenshots

![video1 00_03_35_19 Still005](https://github.com/Churanta/Binary-Brains/assets/83538805/334c7bbb-80b8-42e1-8d5e-e17b75666ee5)

![video1 00_01_14_27 Still004](https://github.com/Churanta/Binary-Brains/assets/83538805/58371187-aacd-4bad-9689-b9bf5e237853)

![video1 00_02_48_19 Still003](https://github.com/Churanta/Binary-Brains/assets/83538805/5851b13c-4757-4d49-a3ad-4506f2abf3da)
