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
* Project Slide: https://docs.google.com/presentation/d/17T4LfeyejFdhVREXjFXpq5B_bg7Bd9_90NFYxr42mG8/edit?pli=1#slide=id.g2cf94e1e45f_0_0

### Precautions
* You are not allowed to modify the pre-defined model structure (gpt2-mini)
* You only have to modify run.py and mingpt/model_multiplier.py to perform experiments.
* Training can take 20-30 mins per run (on GPU).
