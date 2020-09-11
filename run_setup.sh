#!/bin/bash

#### INSTRUCTIONS: Make sure you set your directory to where you want to download the filesand run the script there ####
#### This script will download, extract, and move wav files from the voxforge site ####

# List of languages to download. Based on the URL, add or delete manually if you wish to change languages to scrape
LanguageArray=("en" "fr" "es" "de" "it")

# Sets working directory as where the script was run and where to download the files
cd "$( dirname "${BASH_SOURCE[0]}" )"
echo $(pwd)
rootdir=$(pwd)

#### STEP 1: Download files for each language and wait 5 seconds per download to not overload server #####
for lang in ${LanguageArray[*]}; do

	#### Use wget instead if you want to just download from top to bottom for everything ####
	# -Q limits the max size downloaded per class if you wish to not download the whole repo
	wget -r -np -Q2500m -nc -w3 R "index.html*" -P $rootdir/$lang http://www.repository.voxforge1.org/downloads/$lang/Trunk/Audio/Main/16kHz_16bit/

	#### Use the code below if you want to download a randomized list of files from the csv####
	#[[ ! -d $lang ]] && mkdir $lang; cd $lang;
	# while IFS="," read filename
	# do
	#     curl http://www.repository.voxforge1.org/downloads/$lang/Trunk/Audio/Main/16kHz_16bit/$filename
	# done < "${lang}_list.csv"
	#
	# cd $rootdir;
done


#### STEP 2: Extract files from tgz ####
for lang in ${LanguageArray[*]}; do
	cd $rootdir/$lang/www.repository.voxforge1.org/downloads/$lang/Trunk/Audio/Main/16kHz_16bit;
	for i in *.tgz; do tar -zxvf $i -C $rootdir/$lang; done
done


#### STEP 3: Copy .wav files to corresponding language  ####
for lang in ${LanguageArray[*]}; do
	cd $rootdir;
	# create a new folder for wav files if not exist
	newlang="${lang}_wav";
	[[ ! -d $newlang ]] && mkdir $newlang;
	cd $lang

	# move those files from extracted folders
	for folder in *; do
		# skips the folder with tar files
		if [[ "$folder" == "www.repository.voxforge1.org" ]]; then
			continue
		fi

		# rename and move all the wav files to one subfolder
		if [ -d $rootdir/$lang/$folder/wav ] && [ ! -z "$(ls -A -- $rootdir/$lang/$folder/wav)" ]; then
			cd $rootdir/$lang/$folder/wav;
			for f in *.wav; do mv "$f" "${f}_${folder}.wav"; done
			mv *.wav $rootdir/$newlang
		else
			pass
		fi
	done
done


#### STEP 3: DELETE FOLDERS WE DO NOT NEED ANYMORE ####
# remove all the empty extracted folders we moved the wav files from and all the targz folders we downloaded
cd $rootdir
rm -r en fr de it es
