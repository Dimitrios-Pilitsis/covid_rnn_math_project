#!/bin/bash


# Comment


#countries=('Greece')
#countries=('Greece' 'Italy' 'United Kingdom')
countries=('Greece' 'Italy' 'United_Kingdom' 'United_States' 'Israel')


# Make a directory to store all the models in h5 and txt format respectively
mkdir -p ./models_h5
mkdir -p ./weights

# Make a directory for each country within weights directory to store txt weights
for i in "${countries[@]}"
do
	mkdir -p ./weights/$i	
done


# Run model on each country
for i in "${countries[@]}"
do
	python3 rnn_npi_covid.py $i
done




# Run weight conversion from h5 to txt for each country
for i in "${countries[@]}"
do
	python3 read_write_weights.py $i
done


printf "\n"
