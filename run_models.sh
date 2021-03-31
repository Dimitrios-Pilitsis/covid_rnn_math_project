#!/bin/bash


# Comment


#countries=('Greece')
#countries=('Greece' 'Italy' 'United Kingdom')
countries=('Greece' 'Italy')


for i in "${countries[@]}"
do
	python3 rnn_npi_covid.py $i
done

printf "\n"
