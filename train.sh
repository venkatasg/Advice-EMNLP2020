#!/bin/bash
set -x # echo on

# DEFAULT

python Advice_Classification_Simple.py --data askparents --model bert --multigpu --seed 2
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --seed 13
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --seed 20
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --seed 45
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --seed 78

# QUERY

python Advice_Classification_Simple.py --data askparents --model bert --multigpu --query --seed 2
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --query --seed 13
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --query --seed 20
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --query --seed 45
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --query --seed 78

# CONTEXT

python Advice_Classification_Simple.py --data askparents --model bert --multigpu --context --seed 2
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --context --seed 13
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --context --seed 20
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --context --seed 45
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --context --seed 78

# DEFAULT

python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --seed 2
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --seed 13
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --seed 20
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --seed 45
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --seed 78

# QUERY

python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --query --seed 2
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --query --seed 13
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --query --seed 20
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --query --seed 45
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --query --seed 78

# CONTEXT

python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --context --seed 2
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --context --seed 13
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --context --seed 20
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --context --seed 45
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --context --seed 78

# NO FINETUNING

python Advice_Classification_Simple.py --noft --data askparents --model bert --multigpu --seed 2
python Advice_Classification_Simple.py --noft --data askparents --model bert --multigpu --seed 13
python Advice_Classification_Simple.py --noft --data askparents --model bert --multigpu --seed 20
python Advice_Classification_Simple.py --noft --data askparents --model bert --multigpu --seed 45
python Advice_Classification_Simple.py --noft --data askparents --model bert --multigpu --seed 78

python Advice_Classification_Simple.py --noft --data needadvice --model bert --multigpu --seed 2
python Advice_Classification_Simple.py --noft --data needadvice --model bert --multigpu --seed 13
python Advice_Classification_Simple.py --noft --data needadvice --model bert --multigpu --seed 20
python Advice_Classification_Simple.py --noft --data needadvice --model bert --multigpu --seed 45
python Advice_Classification_Simple.py --noft --data needadvice --model bert --multigpu --seed 78

# TRANSFER

python Advice_Classification_Simple.py --data askparents --model bert --multigpu --seed 2
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --seed 13
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --seed 20
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --seed 45
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --seed 78

python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --seed 2
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --seed 13
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --seed 20
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --seed 45
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --seed 78

python Advice_Classification_Simple.py --data askparents --model bert --multigpu --seed 2
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --seed 13
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --seed 20
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --seed 45
python Advice_Classification_Simple.py --data askparents --model bert --multigpu --seed 78

python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --seed 2
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --seed 13
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --seed 20
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --seed 45
python Advice_Classification_Simple.py --data needadvice --model bert --multigpu --seed 78