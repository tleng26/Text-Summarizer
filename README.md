# Text Summarizer
This project is dedicated to implementing a sequence-to-sequence model using abstractive text summarization techniques. Our goal was to enhance the Google Chrome abstract summarization feature, designed to condense texts such as articles and papers into concise summaries that capture essential information. By integrating machine learning, we aimed to improve the accuracy and relevance of the summaries, ensuring they maintain the core meanings while eliminating extraneous details. This enhancement not only aims to streamline user experience but also to provide a more efficient tool for accessing quick, reliable summaries directly within the browser.




## Preprocessing
We utilized the Amazon Fine Food Reviews dataset to train our model. Our preprocessing involves cleaning the text by removing non-alphabetic characters and numbers, expanding contractions, filtering out stop words, and stemming words to their roots. These steps help standardize the input, improving the model’s learning efficiency.

Due to GitHub's file size limitations, the dataset can be accessed via Kaggle: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews.




## Building the Model
### Text Vectorization
We must tokenize the text and map each word to a unique integer because neural networks process numerical data, not text. Here's an example:

- Input Text: [‘what doing’, ‘how are you’, ‘good’]
- Tokenized Dictionary: {‘what’: 1, ‘doing’: 2, ‘how’: 3, ‘are’: 4, ‘you’: 5, ‘good’: 6}
- So if this is a different input sequence: [‘what are you doing’, ‘you are good’]
- Then this is the sequence vectorized: [[1, 4, 5, 2], [5, 4, 6]

During this process, we must ensure all input and target texts have the same length by padding them with zeros if they are shorter than the maximum length. This uniform length is necessary for efficient processing in batches and is essential for parallel processing and maintaining the structural integrity of the data during training.

- So if the uniform length is 5, then [[1, 4, 5, 2], [5, 4, 6]] becomes [[1, 4, 5, 2, 0], [5, 4, 6, 0, 0]].

### Embedding
We will use an embedding layer to transform each number in the sequence to a dense vector of fixed size. Think of each vector representing a point in some dimensional space, which means that we are representing a word using a point. Each point in space is determined by the semantic of the word, so words that are semantically similar (i.e. king, queen) will be closer to each other in this dimensional space. This allows a numerical representation of words, which will make it possible for our model to process the sequences.

### LSTM Layers
As a basic overview, each LSTM encoder layer will attempt to summarize the input sequence while giving contextual information and will pass only the summary to the next layer. The next encoder layer will use the passed summary to attempt to generate a better summary and give better contextual information.

The single LSTM decoder layer will take the contextual information from the final layer and generate a final summary that will represent the summary of the beginning input text. Think of it like this:

- Imagine a game of telephone where each participant (representing an LSTM encoder layer) refines and summarizes the message based on their understanding and passes the message on to the next person. This process continues until the last person in the encoder chain receives the most refined and condensed version of the original message.
- Now, imagine an outside person, who represents the LSTM decoder layer, listens to the last person’s thoughts about the information and context of the text, and uses this to recreate the full message in a clear and detailed form. This recreated version is the decoder's output.

### Attention Mechanism
After obtaining the summary from the encoder and decoder, we will pass this information to the attention layer. We generate our final summary word by word, and the attention layer creates a mechanism that allows the model to focus on important parts of the input sequence when predicting each word of the output sentence. Then the dense layer is applied, which takes this and generates the probabilities of the next word as the output sequence is being formed. The probabilities will determine the next word to come in the output. Here's an example:

- Input: “AI applications can be found in various fields, including healthcare, education, and finance.”
- Encoder output: The encoder's outputs are important information about each part of this sentence.
- Decoder output: Let’s say that the decoder is attempting to create its output and just generated the word “field”.
- The attention layer looks at both the decoder’s current state (“field”) and the encoder’s outputs. From this, it decides which parts of the encoder's output are most relevant for generating the word after “field”.
- These parts of the input paragraph are outputted into a context vector, and the dense layer takes this and combines the information with the decoder’s output (final summary) to create the probabilities of the next word.

### Training the Model
We split the dataset into training and testing sets in an 80:20 ratio. After building our model using the encoder inputs (input sequence), decoder inputs (target sequence), and decoder outputs (predicted sequences), we compile it and train it over 10 cycles using a batch size of 512 samples each cycle, using 10% of the data to validate our model. The saved model and its variables are in the "s2s" folder, and a visual of the model layers is shown in "model_plot.png".

### Inference Model
We will use our trained model to create an inference architecture for the encoder and decoder model using the same steps and techniques we used for our original model. This inference model is used to test the new sequences for which the target sequence is not known.




## Results
Finally, we can predict the summary for the user input reviews. The results demonstrate the model's ability to generate coherent and contextually accurate summaries, showcasing the effectiveness of the Seq2Seq architecture with attention for text summarization tasks. Here are some examples:

- **Review:** let me first say that i love sour candies, but these worms were way too sour, to the point that you can't really taste much flavor because the sourness overpowers everything. there are only 3 color combos, as seen in the picture (yellow/green, red/yellow, and orange/blue). that's right, there is NO blue/pink combo like the trolli ones, which is quite disappointing.  and even if you put the red half with the blue half, it wouldn't taste like the trolli ones either. in the case of sour worms, albanese &lt; trolli, hands down.

  **Predicted Summary:** too sour

- **Review:** I started buying the sugar free version to cut back on my sugar intake. There's a noticeable difference between the two which is as obvious as a regular coke and a diet coke. I learned to add coffee, bananas, and ice to make it taste better. I noticed that Carnation has significantly cuts back on the amount of powder in each packet; so much so it seems wasteful to continue to use the same packet.

  **Predicted Summary:** not as good as original




## Challenges
- Ensuring that the model architecture was optimized for both performance and resource usage involved a lot of experimentation with different hyperparameters, such as the number of LSTM layers and the learning rate. Overfitting is still an issue with this model, as some summaries are missing key bits of information and/or are too short.
  
- Evaluating the performance of a text summarization model is inherently subjective. Defining clear metrics and benchmarks for assessing the quality of generated summaries posed a challenge. We conducted an assessment by testing our model on 200 reviews, determining that approximately 157 summaries (about 80%) accurately reflected the original content based on our own evaluations.

- The dataset we employed to train our model was not ideally suited for a Google Chrome summarization feature, primarily because most Amazon reviews consist of simple language and do not closely mirror the complexity of professional articles found online. Finding the perfect dataset proved challenging; we required a resource that provided enough examples of texts paired with their precise summaries. Despite its limitations, the Amazon Reviews dataset fulfilled this criterion, making it the most viable option to train our model.




## Future Additions
- Developing a user-friendly interface or an API for the summarization model would make it more accessible for students and developers. This could include a dashboard where users can input text and receive summaries or an API that developers can integrate into their applications to leverage the summarization capabilities.

- Adjusting the length of summaries to correspond with the size of the input text would significantly enhance the effectiveness of our model. We encountered challenges where some summaries were overly concise relative to the depth of information in the original texts. By implementing separate models for longer and shorter texts, we could tailor our summaries more precisely, thereby improving both their accuracy and relevance.




## Usage
To use the Text Summarizer ML Program:
1. Clone the repository: `git clone https://github.com/your-username/Text-Summarizer-ML-Model.git`
2. Install required dependencies: `pip install -r requirements.txt`
3. Download Reviews.csv: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
4. Run the program: `python summarizer_model.py runserver`
5. Follow prompts to interact with program!
