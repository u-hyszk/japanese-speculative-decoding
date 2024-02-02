import os
<<<<<<< HEAD
import gc
=======
>>>>>>> 9b1cecbcd95c79eee31cedae5538b53c0b0a882c
import json
from glob import glob
import argparse
from statistics import mean
from typing import Dict, List, Union, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
<<<<<<< HEAD
from optimum.bettertransformer import BetterTransformer
=======
>>>>>>> 9b1cecbcd95c79eee31cedae5538b53c0b0a882c
from tqdm.auto import tqdm
from sumeval.metrics.rouge import RougeCalculator

from src.decoder import AutoRegressiveDecoder, SpeculativeDecoder
from src.dataloader import get_xlsum_dataloader


def dump_json(
        file: str,
        input: Dict[Any, Any]) -> None:
    with open(file, "w") as f:
        json.dump(input, f, ensure_ascii=False, indent=4)


def load_json(file: str) -> Dict[Any, Any]:
    with open(file, "r") as f:
        output = json.load(f)
    return output


def _decode_logs(
        tokenizer: AutoTokenizer,
        logs: List[Union[str, torch.Tensor, torch.Tensor]]) -> List[str]:
    for i in range(len(logs)):
        for j in range(1, len(logs[i])):
            if logs[i][j] is not None:
                logs[i][j] = tokenizer.decode(logs[i][j][0], skip_special_tokens=False)
    return logs


def _collect_stats(stats_dir: str) -> Dict[str, float]:
    collected_stats = {
        "mean_token_time": [],
        "acceptance_rate": [],
        "rouge_1": [],
        "rouge_2": [],
        "rouge_l": [],
    }
    for stats_file in glob(os.path.join(stats_dir, "stats_*.json")):
        stats = load_json(stats_file)
        for key in stats.keys():
            if key in collected_stats.keys():
                collected_stats[key].append(stats[key])
<<<<<<< HEAD
=======
    print(collected_stats)
>>>>>>> 9b1cecbcd95c79eee31cedae5538b53c0b0a882c
    for key in collected_stats.keys():
        try:
            collected_stats[key] = mean(collected_stats[key])
        except:
            collected_stats[key] = None
    return collected_stats


def parse_arguments():
    parser = argparse.ArgumentParser(description="speculative-decoding")
    parser.add_argument("-i", "--input", type=str, default=None)
<<<<<<< HEAD
    parser.add_argument("-p", "--target_model", type=str, default="u-hyszk/japanese-gpt-neox-409M-xlsum-sft")
    parser.add_argument("-q", "--draft_model", type=str, default="u-hyszk/japanese-gpt-neox-47M-xlsum-sft")
=======
    parser.add_argument("-p", "--target_model", type=str, default="u-hyszk/japanese-gpt-neox-409M-100k-iters-xlsum-finetuned")
    parser.add_argument("-q", "--draft_model", type=str, default="u-hyszk/japanese-gpt-neox-47M-100k-iters-xlsum-finetuned")
>>>>>>> 9b1cecbcd95c79eee31cedae5538b53c0b0a882c
    parser.add_argument("--decode", type=str, default="auto_regressive")
    parser.add_argument("-k", "--n_lookahead", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    # parser.add_argument("--top_k", type=int, default=0)  # TODO
    # parser.add_argument("--top_p", type=float, default=0.0)  # TODO
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_text_length", type=int, default=256)
<<<<<<< HEAD
    parser.add_argument("--bits", type=int, default=32)
    parser.add_argument("--use_better_transformer", action="store_true", default=False)
    parser.add_argument("--use_cache", action="store_true", default=False)
=======
>>>>>>> 9b1cecbcd95c79eee31cedae5538b53c0b0a882c
    return parser.parse_args()


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model & tokenizer
<<<<<<< HEAD
    model_args = {}
    if args.bits == 16:
        model_args["torch_dtype"] = torch.float16
    elif args.bits == 8:
        model_args["load_in_8bit"] = True
    elif args.bits == 4:
        model_args["load_in_4bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if args.use_better_transformer:
        target_model = AutoModelForCausalLM.from_pretrained(args.target_model, device_map="auto", **model_args)
        target_model = BetterTransformer.transform(target_model, keep_original_model=False)
        if args.decode == "speculative":
            draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model, device_map="auto", **model_args)
            draft_model = BetterTransformer.transform(draft_model, keep_original_model=False)
    else:
        target_model = AutoModelForCausalLM.from_pretrained(args.target_model, device_map="auto", **model_args)
        if args.decode == "speculative":
            draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model, device_map="auto", **model_args)
    if args.decode == "speculative":
        assert draft_model.config.vocab_size == target_model.config.vocab_size, "vocab_size must be the same."

=======
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model).to(device)
    if args.decode == "speculative":
        draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model).to(device)
        assert draft_model.config.vocab_size == target_model.config.vocab_size, "vocab_size must be the same."

    # Flash Attention
    # KV-cache
    # ONNX Runtime
    # BetterTransformer
    # torchscript
    # quantization

>>>>>>> 9b1cecbcd95c79eee31cedae5538b53c0b0a882c
    # Get auto_regressive/speculative decoder
    if args.decode == "auto_regressive":
        decoder = AutoRegressiveDecoder(
            target_model=target_model,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
<<<<<<< HEAD
            use_cache=args.use_cache,
=======
>>>>>>> 9b1cecbcd95c79eee31cedae5538b53c0b0a882c
        )
    elif args.decode == "speculative":
        decoder = SpeculativeDecoder(
            target_model=target_model,
            draft_model=draft_model,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
<<<<<<< HEAD
            use_cache=args.use_cache,
=======
>>>>>>> 9b1cecbcd95c79eee31cedae5538b53c0b0a882c
        )
    else:
        raise ValueError("Invalid decode type.")

    # Single decoding
    if args.input:
        print("input:", args.input)
        input_ids = tokenizer.encode(args.input, return_tensors="pt")
        input_ids = input_ids.to(device)
        output_ids = decoder.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            n_lookahead=args.n_lookahead,
            temperature=args.temperature,
        )
        output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        print("output:", output)

    # Benchmarking
    elif args.benchmark:
<<<<<<< HEAD
        print(f"benchmarking...: {args.output_dir}")
=======
        print("benchmarking...")
>>>>>>> 9b1cecbcd95c79eee31cedae5538b53c0b0a882c
        os.makedirs(os.path.join(args.output_dir, "stats"), exist_ok=True)
        dump_json(os.path.join(args.output_dir, "args.json"), args.__dict__)
        rouge = RougeCalculator(stopwords=True, lang="ja")
        dataloader = get_xlsum_dataloader(
            max_text_length=args.max_text_length,
            tokenizer=tokenizer,
        )
        for i, batch in tqdm(enumerate(dataloader)):
            input_ids = tokenizer.encode(batch["inputs"][0], return_tensors="pt").to(device)
            stats = decoder.generate_with_stats(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                n_lookahead=args.n_lookahead,
                temperature=args.temperature,
            )
            stats["input"] = batch["inputs"][0]
            stats["label"] = batch["labels"][0]
            stats["output"] = tokenizer.decode(stats["output_ids"][0], skip_special_tokens=True)[len(stats["input"]):]
            stats["rouge_1"] = rouge.rouge_n(summary=stats["output"], references=stats["label"], n=1)
            stats["rouge_2"] = rouge.rouge_n(summary=stats["output"], references=stats["label"], n=2)
            stats["rouge_l"] = rouge.rouge_l(summary=stats["output"], references=stats["label"])
            if args.decode == "speculative":
                stats["logs"] = _decode_logs(tokenizer, stats["logs"])
            del stats["output_ids"]
            dump_json(os.path.join(args.output_dir, "stats", f"stats_{str(i).zfill(5)}.json"), stats)
<<<<<<< HEAD
            if i >= 10:
                break
=======
>>>>>>> 9b1cecbcd95c79eee31cedae5538b53c0b0a882c
        collected_stats = _collect_stats(os.path.join(args.output_dir, "stats"))
        dump_json(os.path.join(args.output_dir, "stats.json"), collected_stats)
    else:
        raise ValueError("input or benchmark must be specified.")

<<<<<<< HEAD
    # Clean up
    del target_model, decoder
    if args.decode == "speculative":
        del draft_model
    gc.collect()

=======
>>>>>>> 9b1cecbcd95c79eee31cedae5538b53c0b0a882c

if __name__ == "__main__":
    main()