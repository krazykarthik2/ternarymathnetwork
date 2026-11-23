"""
production_slm_1b/dataset_stream.py

Clean, standalone high-throughput streaming dataset for FineWeb-Edu.
Zero dependency on legacy research scripts (mige, hopfield, etc.).
Includes instant remote parquet shard-skipping for fast resume.
"""
import torch
from torch.utils.data import IterableDataset

class FastStreamingDataset(IterableDataset):
    """High-throughput multi-worker streaming dataset with instant shard skipping."""
    def __init__(self, dataset_name, subset, seq_len=512, rank=0, world_size=1, skip_tokens=0):
        self.dataset_name = dataset_name
        self.subset = subset
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.skip_tokens = skip_tokens

    def __iter__(self):
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        worker_info = torch.utils.data.get_worker_info()

        # Calculate exact parquet shard to start from to avoid streaming already-seen files
        # FineWeb-Edu 100BT has 140 shards (~714M tokens each)
        start_shard_idx = 0
        if self.skip_tokens > 0:
            tokens_per_shard = 714_000_000
            start_shard_idx = min(int(self.skip_tokens // tokens_per_shard), 139)

        if self.subset:
            ds = load_dataset(self.dataset_name, name=self.subset, split="train", streaming=True)
        else:
            ds = load_dataset(self.dataset_name, split="train", streaming=True)

        # Skip already-processed parquet shards directly at the dataset container level!
        if start_shard_idx > 0 and hasattr(ds, "_shards"):
            try:
                ds._shards = ds._shards[start_shard_idx:]
            except Exception:
                pass

        # Calculate worker shard index
        if worker_info is not None:
            total_shards = self.world_size * worker_info.num_workers
            shard_id = self.rank * worker_info.num_workers + worker_info.id
        else:
            total_shards = self.world_size
            shard_id = self.rank

        token_buf = []
        for idx, item in enumerate(ds):
            if total_shards > 1 and (idx % total_shards != shard_id):
                continue

            text = item.get("text", "")
            if not text:
                continue
            tokens = tokenizer.encode(text)

            token_buf.extend(tokens)
            while len(token_buf) >= self.seq_len + 1:
                chunk = token_buf[:self.seq_len + 1]
                token_buf = token_buf[self.seq_len:]
                yield (torch.tensor(chunk[:-1], dtype=torch.long),
                       torch.tensor(chunk[1:], dtype=torch.long))
