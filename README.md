# Text Summarizer Machine Learning Model
This GitHub project is dedicated to implementing a many-to-many sequence model using abstractive text summarization techniques. Our goal is to generate concise and informative summaries of user reviews for a product, making product evaluation quick and easy. The model, designed to handle large datasets, will initially be trained and evaluated on the first 100,000 rows from a list of Amazon reviews. By integrating the Attention mechanism, the model will not only focus on specific keywords but also ensure the overall context of the reviews is preserved, enhancing the quality of the generated summaries.

## Dataset
We use the Amazon Fine Food Reviews dataset, which contains 500,000 reviews. For our purposes, we have extracted the first 100,000 rows to train and test our model. The dataset provides a rich source of textual data with both reviews and their corresponding summaries. Due to GitHub's file size limitations, the dataset can be accessed via Kaggle: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews. The dataset structure includes columns such as Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, and Text.

## Preprocessing
Text data often contains noise and inconsistencies, which can hinder model performance. During preprocessing, we clean the text by removing non-alphabetic characters, numbers, and short words. We also expand contractions (e.g., "I'm" to "I am") and filter out stop words (e.g., "the", "a", "an"). Additionally, words are stemmed to their root forms to standardize the text. This step ensures that the data fed into the model is uniform and relevant, enhancing the model's ability to learn meaningful patterns.

## Building the Model
Our model is built using a stacked LSTM (Long Short-Term Memory) architecture with three layers. The encoder processes the input text and captures its context through these layers. Each LSTM layer refines the information, capturing increasingly complex patterns and dependencies. The decoder, initialized with the encoder's final state, generates the summary word by word. An attention mechanism is incorporated to dynamically focus on different parts of the input text during summary generation. This mechanism helps the model to better capture the relevant context needed for accurate summarization.

## Training the Model
We split the dataset into training and testing sets in an 80:20 ratio. The text data is then vectorized into sequences of integers using a tokenizer, and padded to ensure uniform length. The model is compiled with the RMSprop optimizer and sparse categorical crossentropy loss function. Training is performed with a batch size of 512 over 10 epochs, with a validation split of 10% to monitor performance. The trained model is saved for later use in generating summaries for new text inputs.

## Inference and Results
For inference, we construct separate models for the encoder and decoder. The encoder processes the input text to generate context vectors, while the decoder uses these vectors along with the attention mechanism to generate the summary. The attention mechanism allows the decoder to focus on relevant parts of the input text dynamically, improving the quality of the generated summaries. The results demonstrate the model's ability to generate coherent and contextually accurate summaries, showcasing the effectiveness of the Seq2Seq architecture with attention for text summarization tasks.

## Challenges
1. Building and training a stacked LSTM model with an attention mechanism is computationally intensive. Ensuring that the model architecture was optimized for both performance and resource usage involved a lot of experimentation with different hyperparameters, such as the number of LSTM layers and the learning rate. Balancing the depth and complexity of the model to avoid overfitting while still capturing the necessary context and dependencies in the text was a persistent challenge.
2. Evaluating the performance of a text summarization model is inherently subjective. Defining clear metrics and benchmarks for assessing the quality of generated summaries posed a challenge. Balancing between automated evaluation metrics, such as BLEU scores, and manual review to ensure the summaries were contextually accurate and coherent required a thoughtful approach.

## Future Features
1. Implementing real-time summarization capabilities where the model can generate summaries on-the-fly for streaming data or live user inputs could be a valuable addition. This would involve optimizing the model for low-latency inference and possibly deploying it in a scalable cloud environment to handle real-time requests.
2. Developing a user-friendly interface or an API for the summarization model would make it more accessible for students and developers. This could include a web-based dashboard where users can input text and receive summaries or an API that developers can integrate into their applications to leverage the summarization capabilities.

## Usage
To use the Text Summarizer ML Program:
1. Clone the repository: git clone https://github.com/your-username/Text-Summarizer-ML-Model.git
2. Install required dependencies: pip install -r requirements.txt
3. Download Reviews.csv: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
4. Run the program: python summarizer_model.py runserver
5. Follow prompts to interact with program!
