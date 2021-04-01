#!/bin/bash


#countries=('Greece' 'Italy' 'United_Kingdom')

countries=('Aruba' 'Afghanistan' 'Angola' 'Albania' 'Andorra' 'United_Arab_Emirates' 'Argentina' 'Australia' 'Austria' 'Azerbaijan' 'Burundi' 'Belgium' 'Benin' 'Burkina_Faso' 'Bangladesh' 'Bulgaria' 'Bahrain' 'Bahamas' 'Bosnia_and_Herzegovina' 'Belarus' 'Belize' 'Bermuda' 'Bolivia' 'Brazil' 'Barbados' 'Brunei' 'Bhutan' 'Botswana' 'Central_African_Republic' 'Canada' 'Switzerland' 'Chile' 'China' "Cote_d'Ivoire" 'Cameroon' 'Democratic_Republic_of_Congo' 'Congo' 'Colombia' 'Cape_Verde' 'Costa_Rica' 'Cuba' 'Cyprus' 'Czech_Republic' 'Germany' 'Djibouti' 'Dominica' 'Denmark' 'Dominican_Republic' 'Algeria' 'Ecuador' 'Egypt' 'Eritrea' 'Spain' 'Estonia' 'Ethiopia' 'Finland' 'Fiji' 'France' 'Faeroe_Islands' 'Gabon' 'United_Kingdom' 'Georgia' 'Ghana' 'Guinea' 'Gambia' 'Greece' 'Greenland' 'Guatemala' 'Guam' 'Guyana' 'Hong_Kong' 'Honduras' 'Croatia' 'Haiti' 'Hungary' 'Indonesia' 'India' 'Ireland' 'Iran' 'Iraq' 'Iceland' 'Israel' 'Italy' 'Jamaica' 'Jordan' 'Japan' 'Kazakhstan' 'Kenya' 'Kyrgyz_Republic' 'Cambodia' 'Kiribati' 'South_Korea' 'Kuwait' 'Laos' 'Lebanon' 'Liberia' 'Libya' 'Liechtenstein' 'Sri_Lanka' 'Lithuania' 'Luxembourg' 'Latvia' 'Macao' 'Morocco' 'Monaco' 'Moldova' 'Madagascar' 'Mexico' 'Mali' 'Malta' 'Myanmar' 'Mongolia' 'Mozambique' 'Mauritania' 'Mauritius' 'Malaysia' 'Namibia' 'Niger' 'Nigeria' 'Nicaragua' 'Netherlands' 'Norway' 'Nepal' 'New_Zealand' 'Oman' 'Pakistan' 'Panama' 'Peru' 'Philippines' 'Papua_New_Guinea' 'Poland' 'Puerto_Rico' 'Portugal' 'Paraguay' 'Palestine' 'Qatar' 'Kosovo' 'Romania' 'Russia' 'Rwanda' 'Saudi_Arabia' 'Sudan' 'Senegal' 'Singapore' 'Sierra_Leone' 'El_Salvador' 'San_Marino' 'Somalia' 'Serbia' 'Suriname' 'Slovak_Republic' 'Slovenia' 'Sweden' 'Eswatini' 'Seychelles' 'Syria' 'Chad' 'Togo' 'Thailand' 'Turkmenistan' 'Timor-Leste' 'Tonga' 'Trinidad_and_Tobago' 'Tunisia' 'Turkey' 'Taiwan' 'Tanzania' 'Uganda' 'Ukraine' 'Uruguay' 'United_States' 'Uzbekistan' 'Venezuela' 'United_States_Virgin_Islands' 'Vietnam' 'South_Africa' 'Zambia' 'Zimbabwe')



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

