#!/bin/bash

cd ..
cd data

filename_100_twitter_cbow="full_grams_cbow_100_twitter.zip"
mdlFilename_100_twitter_cbow="full_grams_cbow_100_twitter.mdl"
npyTrainablesFilename_100_twitter_cbow="full_grams_cbow_100_twitter.mdl.trainables.syn1neg.npy"
npyVectorsFilename_100_twitter_cbow="full_grams_cbow_100_twitter.mdl.wv.vectors.npy"

# ---------------------------------------------------------------------------

# Download full_grams_cbow_100_twitter.zip if not present on system
if [[ ! -f $filename_100_twitter_cbow ]]
then
	echo "$filename_100_twitter_cbow does not exist."
	echo; echo 
	echo "Downloading $filename_100_twitter_cbow now ... π"
	echo 
	wget https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_cbow_100_twitter.zip
	echo "Download $filename_100_twitter_cbow complete β"
	echo
	echo "Unzipping $filename_100_twitter_cbow ... π"
	python3 pyunzip.py full_grams_cbow_100_twitter.zip
	echo "Unzipping $filename_100_twitter_cbow Complete β"; echo

# Unzip the full_grams_cbow_100_twitter.zip file
elif [[ -f $filename_100_twitter_cbow ]] && [[ ! -f $mdlFilename_100_twitter_cbow ]] && [[ ! -f $npyTrainablesFilename_100_twitter_cbow ]] && [[ ! -f $npyVectorsFilename_100_twitter_cbow ]]
then
	echo "Unzipping $filename_100_twitter_cbow ... π"
	echo
	python3 pyunzip.py full_grams_cbow_100_twitter.zip
	echo "Unzipping $filename_100_twitter_cbow Complete β"; echo

else
	echo "$filename_100_twitter_cbow exists and unzipped. All good!! π"

fi 

# ----------------------------------------------------------------------------


echo "Downloading AraVec models complete! πΎπΎπΎ"; echo

exit 0
