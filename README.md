<h3 align="center">
  <img src="assets/text_predictor_icon_web.png" width="300">
</h3>

# Text Predictor
1. STOPPED reuters 300
2. STOPPED hackernews 100
3. STOPPED war_and_peace 100
4. 
5. reuters 512
6. wikipedia 512
7. f war_and_peace 512
8. f hackernews 512


https://github.com/karpathy/randomfun/blob/master/min-char-rnn-nb3.ipynb

## Datasets
	reuters (580 MB) - collection of reuters headlines
	war_and_peace (3 MB) - Leo Tolstoy's War and Peace novel
	wikipedia (100 MB) - excerpt from English wikipedia
	hackernews (90 KB) - collection of hackernews headlines
	

## Training
Training time: 5 M iterations


### Hyperparameters
	HIDDEN_LAYER_SIZE = 512
	SEQUENCE_LENGTH = 20
	LEARNING_RATE = 1e-1
	ADAGRAD_UPDATE_RATE = 1e-8
	LOGGING_FREQUENCY = 1e3
	TEXT_SAMPLING_FREQUENCY = 1e4
	GRADIENT_LIMIT = 5
	RANDOM_WEIGHT_INIT_FACTOR = 1e-2
	LOSS_SMOOTHING_FACTOR = 0.999
	TEXT_SAMPLE_LENGTH = 200



	
	


https://github.com/karpathy/char-rnn
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

https://www.youtube.com/watch?v=UNmqTiOnRfg
great vid

NEURAL NET
sun -> basketball
rain -> book

RECCURENT NEURAL NET
every other day, diffetent activity
basketball -> book -> basketball -> book

monitor learning progress
