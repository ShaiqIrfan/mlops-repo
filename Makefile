setup:
	python -m pip install -r requirements.txt

download-data:
	python src/download.py

preprocess: download-data
	python src/preprocess.py

features: preprocess
	python src/features.py

train: features
	python src/train.py

predict: train
	python src/predict.py

evaluate: predict
	python src/evaluate.py

all: setup download-data preprocess features train predict evaluate

clean:
	rm -rf data/processed/*
	rm -rf features/*
	rm -rf models/*
	rm -rf result/*
