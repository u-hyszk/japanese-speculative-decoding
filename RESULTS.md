# Results

## 1. Speed-up in Japanese XLSum

- Target model: [japanese-gpt-neox-409M-xlsum-sft](https://huggingface.co/u-hyszk/japanese-gpt-neox-409M-xlsum-sft)
- Draft model: `japanese-gpt-neox-XXX-xlsum-sft` (6M ~ 247M)
- Dataset For Benchmarking: [Test data of Japanese XLSum (889 rows)](https://huggingface.co/datasets/csebuetnlp/xlsum/viewer/japanese/test)
- GPU: RTX 3090 x 1
- Average of 5 repeated runs

| # of draft params | n_lookahead | mean acceptance rate | inference speed rate | Speed up       |
|-------------------|-------------|----------------------|----------------------|----------------|
| 6M                | 1           | 0.393                | 0.052                | x1.32          |
|                   | 3           | 0.280                | 0.052                | x1.19          |
|                   | 5           | 0.280                | 0.052                | x1.09          |
|                   | 7           | 0.280                | 0.052                | x1.02          |
| 13M               | 1           | 0.436                | 0.057                | x1.35          |
|                   | 3           | 0.302                | 0.057                | x1.21          |
|                   | 5           | 0.302                | 0.057                | x1.11          |
|                   | 7           | 0.302                | 0.057                | x1.02          |
| 29M               | 1           | 0.593                | 0.124                | <span style="color: red;">x1.42</span>          |
|                   | 3           | 0.373                | 0.124                | x1.15          |
|                   | 5           | 0.373                | 0.124                | x0.98          |
|                   | 7           | 0.373                | 0.124                | x0.85          |
| 47M               | 1           | 0.620                | 0.164                | x1.39          |
|                   | 3           | 0.384                | 0.164                | x1.08          |
|                   | 5           | 0.384                | 0.164                | x0.89          |
|                   | 7           | 0.384                | 0.164                | x0.76          |
| 72M               | 1           | 0.651                | 0.229                | x1.35          |
|                   | 3           | 0.396                | 0.229                | x0.97          |
|                   | 5           | 0.396                | 0.229                | x0.77          |
|                   | 7           | 0.396                | 0.229                | x0.64          |
| 115M              | 1           | 0.671                | 0.358                | x1.23          |
|                   | 3           | 0.403                | 0.358                | x0.80          |
|                   | 5           | 0.403                | 0.358                | x0.60          |
|                   | 7           | 0.403                | 0.358                | x0.47          |
| 165M              | 1           | 0.672                | 0.426                | x1.16          |
|                   | 3           | 0.403                | 0.426                | x0.72          |
|                   | 5           | 0.403                | 0.426                | x0.52          |
|                   | 7           | 0.403                | 0.426                | x0.41          |
| 247M              | 1           | 0.679                | 0.622                | x1.02          |
|                   | 3           | 0.407                | 0.622                | x0.57          |
|                   | 5           | 0.407                | 0.622                | x0.40          |
|                   | 7           | 0.407                | 0.622                | x0.31          |
| 407M              | -           | -                    | -                    | -              |

## 2. Speed-up by combining methods

- Target model: [japanese-gpt-neox-409M-xlsum-sft](https://huggingface.co/u-hyszk/japanese-gpt-neox-409M-xlsum-sft)
- Draft model: [japanese-gpt-neox-29M-xlsum-sft](https://huggingface.co/u-hyszk/japanese-gpt-neox-29M-xlsum-sft)
- n_lookahead: 1
- Dataset For Benchmarking: [Test data of Japanese XLSum (889 rows)](https://huggingface.co/datasets/csebuetnlp/xlsum/viewer/japanese/test)
- GPU: RTX 3090 x 1
- Average of 5 repeated runs

| Methods      |              |              | Speed up  |
|--------------|--------------|--------------|-------------------|
| Speculative Decoding | KV-cache | Better Transformer |         |
| $\times$     | $\times$     | $\times$     | -                 |
| $\times$     | $\checkmark$ | $\times$     | x1.39             |
| $\times$     | $\times$     | $\checkmark$ | x1.12             |
| $\times$     | $\checkmark$ | $\checkmark$ | x1.53             |
| $\checkmark$ | $\times$     | $\times$     | x1.41             |
| $\checkmark$ | $\checkmark$ | $\times$     | <span style="color: red;">x1.77</span>|
| $\checkmark$ | $\times$     | $\checkmark$ | x1.60             |
| $\checkmark$ | $\checkmark$ | $\checkmark$ | x1.60             |
