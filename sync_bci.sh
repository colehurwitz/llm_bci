#!/bin/bash
rsync -r -u --progress  /home/gridsan/dbeneto/TFG/BCI/ llm4bci@18.223.86.50:/home/llm4bci/llm_bci/ --exclude .git --exclude .gitignore --exclude __pycache__ --exclude "*.sh" 
