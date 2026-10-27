#!/bin/bash

python whisper_onecase_vad.py --th 1.5

#for i in {1.5,2.5}
#do 
    #echo "Running threshold $i"
    #python whisper_onecase_vad.py --th $i
#done