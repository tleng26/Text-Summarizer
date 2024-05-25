# Introduction
This project is dedicated to implementing a many-to-many sequence model using abstractive text summarization techniques. Our goal is to generate concise and informative summaries of user reviews for a product, making product evaluation quick and easy.

# Preprocessing
Amazon Fine Food Reviews dataset, which provides a rich source of textual data with both reviews and their corresponding summaries. Due to GitHub's file size limitations, the dataset can be accessed via Kaggle: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews. We will be extracting the "Summary" and "Text" columns from this dataset to build our model.

During preprocessing, we clean the text by removing non-alphabetic characters, numbers, and short words. We also expand contractions (e.g., "I'm" to "I am") and filter out stop words (e.g., "the", "a", "an"). Additionally, words are stemmed to their root forms to standardize the text. This step ensures that the data fed into the model is uniform and relevant, enhancing the model's ability to learn meaningful patterns.

# Building the Model
### Text Vectorization
We must tokenize the text and map each word to a unique integer because neural networks process numerical data, not text. Here's an example:

Input Text: [‘what doing’, ‘how are you’, ‘good’]

Tokenized Dictionary: {‘what’: 1, ‘doing’: 2, ‘how’: 3, ‘are’: 4, ‘you’: 5, ‘good’: 6}

So if this is a different input sequence: [‘what are you doing’, ‘you are good’]

Then this is the sequence vectorized: [[1, 4, 5, 2], [5, 4, 6]

During this process, we must ensure all input and target texts have the same length by padding them with zeros if they are shorter than the maximum length. This uniform length is necessary for efficient processing in batches and is essential for parallel processing and maintaining the structural integrity of the data during training.

So if the uniform length is 5, then [[1, 4, 5, 2], [5, 4, 6]] becomes [[1, 4, 5, 2, 0], [5, 4, 6, 0, 0]].

### Embedding
We will use an embedding layer to transform each number in the sequence to a dense vector of fixed size. Think of each vector representing a point in some dimensional space, which means that we are representing a word using a point. Each point in space is determined by the semantic of the word, so words that are semantically similar (i.e. king, queen) will be closer to each other in this dimensional space. This allows a numerical representation of words, which will make it possible for our model to process the sequences.

### LSTM Layers
As a basic overview, each LSTM encoder layer will attempt to summarize the input sequence while giving contextual information and will pass only the summary to the next layer. The next encoder layer will use the passed summary to attempt to generate a better summary and give better contextual information.

The single LSTM decoder layer will take the contextual information from the final layer and generate a final summary that will represent the summary of the beginning input text. Think of it like this:

- Imagine a game of telephone where each participant (representing an LSTM encoder layer) not only passes along the message they heard but also refines and summarizes it based on their understanding. Each person listens to the message, condenses it into a more concise and clear version while retaining its core meaning, and then whispers it to the next person. This process continues until the last person in the encoder chain receives the most refined and condensed version of the original message.
- Now, imagine a final person, who represents the LSTM decoder layer, listens to the last person’s thoughts about the information and context of the text, and uses this to recreate the full message in a new, clear, and detailed form. This recreated version, which highlights the main points and context emphasized through the summaries, is the decoder's output.

### Attention Mechanism
After obtaining the summary from the encoder and decoder, we will pass this information to the attention layer. We generate our final summary word by word, and the attention layer creates a mechanism that allows the model to focus on important parts of the input sequence when predicting each word of the output sentence. In simple terms, the mechanism essentially decides what parts of the input paragraph are relevant for generating the next word in the decoder output sequence. Then the dense layer is applied, which takes this and generates the probabilities of the next word as the output sequence is being formed. The probabilities will determine the next word to come in the output. Here's an example:

- Input: “AI applications can be found in various fields, including healthcare, education, and finance.”
- Encoder output: The encoder's outputs are important information about each part of this sentence.
- Decoder output: Let’s say that the decoder is attempting to create its output and just generated the word “field”.
- The attention layer looks at both the decoder’s current state (“field”) and the encoder’s outputs. From this, it decides which parts of the encoder's output are most relevant for generating the word after “field”.
- These parts of the input paragraph are outputted into a context vector, and the dense layer takes this and combines the information with the decoder’s output (final summary) to create the probabilities of the next word.

# Training the Model
We split the dataset into training and testing sets in an 80:20 ratio. After building our model using the encoder inputs (input sequence), decoder inputs (target sequence), and decoder outputs (predicted sequences), we compile it and train it over 10 cycles using a batch size of 512 samples each cycle, using 10% of the data as validating our model. The saved model and its variables are in the "s2s" folder, and a visual of the model layers is shown in "model_plot.png".

# Inference and Results
For inference, we construct separate models for the encoder and decoder. The encoder processes the input text to generate context vectors, while the decoder uses these vectors along with the attention mechanism to generate the summary. The attention mechanism allows the decoder to focus on relevant parts of the input text dynamically, improving the quality of the generated summaries. The results demonstrate the model's ability to generate coherent and contextually accurate summaries, showcasing the effectiveness of the Seq2Seq architecture with attention for text summarization tasks.

# Challenges
- Building and training a stacked LSTM model with an attention mechanism is computationally intensive. Ensuring that the model architecture was optimized for both performance and resource usage involved a lot of experimentation with different hyperparameters, such as the number of LSTM layers and the learning rate. Balancing the depth and complexity of the model to avoid overfitting while still capturing the necessary context and dependencies in the text was a persistent challenge.
- Evaluating the performance of a text summarization model is inherently subjective. Defining clear metrics and benchmarks for assessing the quality of generated summaries posed a challenge. Balancing between automated evaluation metrics, such as BLEU scores, and manual review to ensure the summaries were contextually accurate and coherent required a thoughtful approach.

# Future Features
- Implementing real-time summarization capabilities where the model can generate summaries on-the-fly for streaming data or live user inputs could be a valuable addition. This would involve optimizing the model for low-latency inference and possibly deploying it in a scalable cloud environment to handle real-time requests.
- Developing a user-friendly interface or an API for the summarization model would make it more accessible for students and developers. This could include a web-based dashboard where users can input text and receive summaries or an API that developers can integrate into their applications to leverage the summarization capabilities.

# Usage
To use the Text Summarizer ML Program:
1. Clone the repository: git clone https://github.com/your-username/Text-Summarizer-ML-Model.git
2. Install required dependencies: pip install -r requirements.txt
3. Download Reviews.csv: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
4. Run the program: python summarizer_model.py runserver
5. Follow prompts to interact with program!
