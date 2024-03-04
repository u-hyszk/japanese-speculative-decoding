"""dataloader for japanese xlsum dataset"""

from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def _sample(
        dataset: Dataset,
        n_samples: int = None,
        seed: int = None) -> Dataset:
    dataset = dataset.shuffle(seed=seed)
    return dataset.select(range(n_samples))


def _truncate(
        text: str,
        tokenizer: AutoTokenizer = None,
        max_length: int = None) -> str:
    if tokenizer is None:
        return text[:max_length]
    else:
        return tokenizer.decode(
            tokenizer.encode(text, max_length=max_length),
            skip_special_tokens=True,
        )


def _get_dataloader(
        dataset: Dataset,
        batch_size: int = 1,
        num_workers: int = 0) -> DataLoader:
    dataset.set_format(type="torch", columns=["inputs", "labels"])
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def get_xlsum_dataloader(
        batch_size: int = 1,
        num_workers: int = 0,
        n_samples: int = None,
        max_text_length: int = None,
        tokenizer: AutoTokenizer = None,
        seed: int = 0,
        **kwargs) -> DataLoader:
    """get pytorch dataloader for japanese xlsum dataset

    each batch format is as follows:
    {
        "inputs": "タイトル: {title}\nテキスト: {text}\n要約: ",
        "labels": {summary},
    }

    for more information,
    see https://huggingface.co/datasets/csebuetnlp/xlsum/viewer/japanese/test

    Parameters
    ----------
    batch_size : int, optional
        pytorch dataloader batch_size (strictly 1 now), by default 1
    num_workers : int, optional
        pytorch dataloader num_workers, by default 0
    n_samples : int, optional
        number of samples to use, by default None
    max_text_length : int, optional
        max length of text to use in xlsum, by default None
    tokenizer : AutoTokenizer, optional
        tokenizer to use for truncating text, by default None
    seed : int, optional
        random seed for sampling dataset, by default 0

    Returns
    -------
    DataLoader
        pytorch dataloader for japanese xlsum dataset
    """
    ds = load_dataset("csebuetnlp/xlsum", name="japanese", split="test")
    if n_samples is not None:
        ds = _sample(ds, n_samples=n_samples, seed=seed)
    ds = ds.map(
        lambda example: {
            "inputs":
                f"タイトル: {example['title']}\n"
                f"テキスト: {_truncate(example['text'], tokenizer=tokenizer, max_length=max_text_length)}\n"
                "要約: ",
            "labels": example["summary"],
        },
    )
    return _get_dataloader(ds, batch_size, num_workers)
