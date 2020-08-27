#!/bin/bash

#### NOTE: Make sure you set your directory to where you want to download the files before running script ####
#### This script creates the data directory and downloads plus extracts all raw wav files in the following langauges ####
LanguageArray=("en" "fr" "es" "de")

# creates a new folder in your current working directory
[[ ! -d vox_lang ]] && mkdir vox_lang;
cd vox_lang;
rootdir=$(pwd)

# download 1gb for each language and wait 5 seconds per download to not overload server
for lang in ${LanguageArray[*]}; do
	# Use wget instead if you want to just download from top to bottom or everything (remove the -Q limit if so)
	# wget -r -np -Q1024m -nc -w3 R "index.html*" -P /$lang http://www.repository.voxforge1.org/downloads/$lang/Trunk/Audio/Main/16kHz_16bit/

	# create folder for each language and change to that directory
	[[ ! -d $lang ]] && mkdir $lang; cd $lang;

	# download each file on the list
	while IFS="," read filename
	do
	    curl http://www.repository.voxforge1.org/downloads/$lang/Trunk/Audio/Main/16kHz_16bit/$filename
	done < "${lang}_list.csv"

	cd $rootdir;
done

# extract files from tgz
for lang in ${LanguageArray[*]}; do
	cd $rootdir/$lang/www.repository.voxforge1.org/downloads/$lang/Trunk/Audio/Main/16kHz_16bit;
	for i in *.tgz; do tar -zxvf $i -C $rootdir/$lang; done
done

# copy .wav files to corresponding language
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
			echo "if worked and found www.repo folder"
		fi

		# rename and move all the wav files to one subfolder
		cd $rootdir/$lang/$folder/wav;
		for f in *.wav; do mv "$f" "${f}_${folder}.wav";done
		mv *.wav $rootdir/$newlang
	done
done
