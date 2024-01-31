from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def _sample(
    dataset: Dataset,
    n_samples: int = None,
    seed: int = None,
) -> Dataset:
    dataset = dataset.shuffle(seed=seed)
    return dataset.select(range(n_samples))


def _truncate(
    text: str,
    tokenizer: AutoTokenizer = None,
    max_length: int = None,
):
    if tokenizer is None:
        return text[:max_length]
    else:
        return tokenizer.decode(tokenizer.encode(text, max_length=max_length), skip_special_tokens=True)


def _get_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    dataset.set_format(type="torch", columns=["inputs", "labels"])
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def get_xlsum_dataloader(
    batch_size: int = 1,
    num_workers: int = 0,
    n_samples: int = None,
    max_text_length: int = None,
    tokenizer: AutoTokenizer = None,
    seed: int = 0,
    **args,
):
    """_summary_

    Parameters
    ----------
    batch_size : int, optional
        _description_, by default 1
    num_workers : int, optional
        _description_, by default 0
    n_samples : int, optional
        _description_, by default None
    max_text_length : int, optional
        _description_, by default None
    tokenizer : AutoTokenizer, optional
        _description_, by default None
    seed : int, optional
        _description_, by default 0

    Returns
    -------
    _type_
        _description_
    """
    ds = load_dataset("csebuetnlp/xlsum", name="japanese", split="test")
    if n_samples is not None:
        ds = _sample(ds, n_samples=n_samples, seed=seed)
    ds = ds.map(
        lambda example: {
            "inputs": "タイトル: " + example["title"] + "\n" + \
                "テキスト: " + _truncate(example["text"], tokenizer=tokenizer, max_length=max_text_length) + "\n" + \
                "要約: ",
            "labels": example["summary"],
        },
    )
    return _get_dataloader(ds, batch_size, num_workers)