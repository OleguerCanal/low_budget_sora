# Low-budget SORA


Just me playing around with video generation.

### Editable install

From the repo root:

```bash
pip install -e .
```

### Usage (Python)

```python
from low_budget_sora import make_sliding_digits, save_gif

clip = make_sliding_digits([1, 2, 3, 4], T=16, H=32, W=32, seed=0)
save_gif("digits.gif", clip, duration=0.1)
```