#!/bin/bash

cd miniature-potato

pip install -r requirements.txt

git config --global credential.helper store

apt-get install git-lfs
git lfs install

FILE="hf.txt"
if [ -e $FILE ]
then
    echo "File $FILE exists"
else
    read -s -p "Upload hf.txt: " ignore
fi
mkdir ~/.huggingface
cp hf.txt ~/.huggingface/token

FILE="kaggle.json"
if [ -e "~/.kaggle/$FILE" ]
then
    echo "File $FILE exists"
else
    
    if [ -e $FILE ]
    then
        echo "copying $FILE to ~/.kaggle"
    else
        read -s -p "Upload kaggle.json: " ignore
    fi
    
    mkdir ~/.kaggle
    
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json

    mkdir data
    kaggle competitions download -c AI4Code
    unzip -q AI4Code.zip -d data
fi

FILE="wandb.txt"
if [ -e $FILE ]
then
    echo "File $FILE exists"
else
    read -s -p "Upload wandb.txt: " ignore
fi

export WANDB_API_KEY=$( cat wandb.txt )
wandb login

export GIT_EMAIL=$( cat email.txt )
export GIT_NAME=$( cat name.txt )

git config --global user.email "$GIT_EMAIL"
git config --global user.name "$GIT_NAME"