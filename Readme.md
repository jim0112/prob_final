# 2024 Prob Final Project

### Code Tree Structure
```
prob_final
├── ChickenRabbit.py (making 雞兔同籠 dataset and evaluation method)
├── GCD.py (making gcd dataset and evaluation method)
├── mingpt
│   ├── bpe.py
│   ├── __init__.py
│   ├── model_multiplier.py (define model)
│   ├── trainer_multiplier.py (define trainer)
│   └── utils.py (some helper functions)
├── Readme.md
└── run.py (execute the whole training logic)
```
### Main Purpose
* Using *hypothsis test* and *p-value* to identify the effect of different **model weight initialization** and **data order** on training iterations.
* Project Slide: https://docs.google.com/presentation/d/1qGTQq95Rn6JPsNvY6nwrSqlAG5rEFn9_WHgGpItVlpQ/edit#slide=id.g2cf5ce42891_2_19

### Precautions
* You are not allowed to modify the pre-defined model structure (gpt2-mini)
* You only have to modify run.py and mingpt/model_multiplier.py to perform experiments.
* Training can take 20-30 mins per run (on GPU).
