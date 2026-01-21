import torch
from torch.utils.data import DataLoader, Dataset

class SimpleTextDataset(Dataset):
    """简单的文本数据集示例"""
    def __init__(self, num_samples: int = 1000, seq_len: int = 128, vocab_size: int = 30000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 模拟数据：随机生成token ids
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        attention_mask = torch.ones(self.seq_len)
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
