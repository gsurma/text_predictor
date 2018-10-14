<h3 align="center">
  <img src="assets/text_predictor_icon_web.png" width="300">
</h3>

# Text Predictor
Character-level RNN (Recurrent Neural Net) LSTM implemented in Python 2.7/tensorflow in order to predict a text based on a given dataset. 

Heavily influenced by [http://karpathy.github.io/2015/05/21/rnn-effectiveness/]().

## Idea
1. Train Recurrent Neural Net LSTM on a given dataset (.txt file).
2. Predict text based on a trained model.


## Usage
1. Clone the repo.
2. Go to the project's root folder.
3. Install required packages`pip install -r requirements.txt`.
4. `python text_predictor.py <dataset>`.

## Datasets
	reuters - a collection of reuters headlines (580 MB)
	war_and_peace - Leo Tolstoy's War and Peace novel (3 MB)
	wikipedia - excerpt from English Wikipedia (100 MB) 
	hackernews - a collection of Hackernews headlines (90 KB)
	sherlock - a collection of Sherlock Holmes books (3 MB)
	shakespeare - the complete works of William Shakespeare (4 MB)
Feel free to add new datasets. Just create a folder in the `./data` directory and put an `input.txt` file there. Output file along with the training plot will be automatically generated there.
	

## Results

TODOEach dataset were trained for 5 M iterations (about 80 hours on 2.9 GHz Intel i7 Quad-Core CPU) with the same hyperparameters.

**Hyperparameters**

	BATCH_SIZE = 32
	SEQUENCE_LENGTH = 50
	LEARNING_RATE = 0.01
	DECAY_RATE = 0.97
	HIDDEN_LAYER_SIZE = 256
	CELLS_SIZE = 2

### Sherlock
<img src="data/sherlock/loss.png" width="500">


### Shakespeare
<img src="data/shakespeare/loss.png" width="500">


### Wikipedia
<img src="data/wikipedia/loss.png" width="500">


### Reuters
<img src="data/reuters/loss.png" width="500">


### Hackernews
<img src="data/hackernews/loss.png" width="500">


### War and Peace
<img src="data/war_and_peace/loss.png" width="500">


<br> 

