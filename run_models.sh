#!/bin/bash


# Comment


#countries=('Greece')
countries=('Greece' 'Italy' 'United Kingdom')
echo ${countries[0]}
#python3 rnn_npi_covid.py

#for (( counter=10; counter>0; counter-- ))
for i in "${countries[@]}"
do
	echo $i
done
printf "\n"
