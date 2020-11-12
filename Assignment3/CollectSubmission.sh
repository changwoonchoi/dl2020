files="./models/*
Assignment3_Part1_Implementing_RNN.ipynb
rnn_layers.py
Assignment3_Part2_Char_RNN.ipynb
char_rnn.py
Assignment3_Part3_Transformer.ipynb
transformer_modules.py
"


for file in $files
do
    if [ ! -f $file ]; then
        echo "Required $file not found."
        exit 0
    fi
done

if [ -z "$1" ]; then
    echo "Student number is required.
Usage: ./CollectSubmission 20xx_xxxxx"
    exit 0
fi


rm -f $1.zip
zip -r $1.zip $files
