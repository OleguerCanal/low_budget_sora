import random
from typing import Literal, Optional

class SequenceSampler:
    CHARS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    DEFAULT_CHARS = CHARS + NUMBERS
    
    def __init__(self,
            num_training_examples: int = 10_000,
            num_validation_examples: int = 1_000,
            min_len: int = 1,
            max_len: int = 6,
            chars: list[str] = DEFAULT_CHARS,
            seed: Optional[int] = 0,
        ):
        self.num_training_examples = num_training_examples
        
        self.min_len, self.max_len = min_len, max_len
        self.chars = chars

        self._val_rng = random.Random(seed)
        self._train_rng = random.Random(seed + 1)
        self._create_val_dataset(n=num_validation_examples)
        

    def _create_val_dataset(self, n: int) -> list[list[str]]:
        """ Create n unique sequences
        """
        self.val_set, self.val_list = set(), []
        while len(self.val_set) < n:
            s = self.sample(self._val_rng)
            if s not in self.val_set:
                self.val_set.add(s)
                self.val_list.append(s)

    def sample(self, rng, n_elems: int = None) -> str:
        if n_elems is None:
            n_elems = rng.randint(self.min_len, self.max_len)
        elems = rng.choices(self.chars, k=n_elems)
        return "".join(elems)
    
    def tokenize(self, sequence: str) -> list[int]:
        # + 1 because we want to reserve 0 for the padding token
        return [self.chars.index(char) + 1 for char in sequence]
    
    def detokenize(self, tokens: list[int]) -> str:
        # - 1 because we want to remove the padding token
        return "".join([self.chars[token - 1] for token in tokens])

    def _format(self, sequence: str) -> list[int]:
        return {
            "sequence": sequence,
            "tokens": self.tokenize(sequence),
            "length": len(sequence),
        }

    def __len__(self) -> int:
        return self.num_training_examples

    def __getitem__(self, idx: int, split: Literal["train", "val"] = "train") -> str:
        if split == "train":
            sample = self.sample(self._train_rng)
            while sample in self.val_set:
                sample = self.sample(self._train_rng)
            return self._format(sample)
                
        if split == "val":
            return self._format(self.val_list[idx])
        
        raise ValueError(f"Invalid split: {split}")

if __name__ == "__main__":
    import time
    time_start = time.time()
    sampler = SequenceSampler(
        num_training_examples=100_000,
        num_validation_examples=1_000,
        seed=42
    )
    time_end = time.time()

    print(f"Time taken: {time_end - time_start} seconds")
    print(f"Number of training examples: {len(sampler)}")
    print(f"Number of validation examples: {len(sampler.val_list)}")
    
    print("Training examples:")
    te = [sampler[i, "train"] for i in range(30)]
    print(te)
    
    print("\nValidation examples:")
    ve = [sampler[i, "val"] for i in range(30)]
    print(ve)
    
    # print(l en(sampler.DEFAULT_CHARS))