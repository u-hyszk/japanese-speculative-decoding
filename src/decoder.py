"""Decoder module for text generation"""

from typing import Dict, List, Union, Any
import time

import torch
import torch.nn.functional as F


class BaseDecoder:
    """Base class for decoders
    """
    def __init__(
            self,
            target_model: torch.nn.Module,
            eos_token_id: int = None,
            pad_token_id: int = None,
            use_cache: bool = False,
            **kwargs) -> None:
        """Initialize the decoder

        Parameters
        ----------
        target_model : torch.nn.Module
            target model to generate text
        eos_token_id : int, optional
            end of sentence token id, by default None
        pad_token_id : int, optional
            padding token id, by default None
        use_cache : bool, optional
            use kv-cache for fast text generation, by default False
        """
        self.target_model = target_model
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_cache = use_cache

    def _is_terminator(self, token_id: torch.Tensor) -> bool:
        """Check if the token is a terminator (= EOS or PAD)

        Parameters
        ----------
        token_id : torch.Tensor
            token id

        Returns
        -------
        bool
            True if the token is a terminator, False otherwise
        """
        if self.pad_token_id is not None and token_id == self.pad_token_id:
            return True
        elif self.eos_token_id is not None and token_id == self.eos_token_id:
            return True
        else:
            return False

    def _logits2probs(
            self,
            logits: torch.Tensor,
            temperature: float = 1.) -> torch.Tensor:
        """Convert logits to probability distribution

        Parameters
        ----------
        logits : torch.Tensor
            raw model logits
        temperature : float, optional
            temperature for sampling tokens from
            the probability distribution, by default 1.

        Returns
        -------
        torch.Tensor
            probability distribution
        """
        # TODO: top_k, top_p filtering
        # temperature adjustment
        logits = logits / (temperature + torch.finfo(logits.dtype).eps)
        # get probability distribution (sum of probs is equal to 1)
        probs = F.softmax(logits, dim=-1)
        return probs

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            **kwargs) -> torch.Tensor:
        """override this method to implement text generation

        generate text from the input_ids

        Parameters
        ----------
        input_ids : torch.Tensor
            input token ids

        Returns
        -------
        torch.Tensor
            generated token ids
        """
        pass

    @torch.no_grad()
    def generate_with_stats(
            self,
            input_ids: torch.Tensor,
            **kwargs) -> Dict[str, Any]:
        """override this method to implement text generation with stats

        generate text from the input_ids and return stats

        Parameters
        ----------
        input_ids : torch.Tensor
            input token ids

        Returns
        -------
        Dict[str, Any]
            stats with generated token ids
        """
        pass


class AutoRegressiveDecoder(BaseDecoder):
    """Auto-regressive decoding
    """
    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 30,
            temperature: float = 1.,
            **kwargs) -> torch.Tensor:
        """Generate text auto-regressively

        Parameters
        ----------
        input_ids : torch.Tensor
            input token ids
        max_new_tokens : int, optional
            maximum number of tokens to generate, by default 30
        temperature : float, optional
            temperature for sampling tokens from, by default 1.

        Returns
        -------
        torch.Tensor
            generated token ids
        """
        past_key_values = None
        for _ in range(max_new_tokens):
            if self._is_terminator(input_ids[:, -1]):
                break
            if self.use_cache:
                if past_key_values is None:
                    past_key_values = self.target_model(
                        input_ids[:, :-1],
                        use_cache=True,
                    ).past_key_values
                model_inputs = self.target_model.prepare_inputs_for_generation(
                    input_ids,
                    past_key_values=past_key_values,
                )
                outputs = self.target_model(**model_inputs, use_cache=True)
                next_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
            else:
                next_logits = self.target_model(
                    input_ids,
                    use_cache=False,
                ).logits[:, -1, :]
            probs = self._logits2probs(next_logits, temperature=temperature)
            token_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, token_id), dim=1)
        return input_ids

    @torch.no_grad()
    def generate_with_stats(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 30,
            temperature: float = 1.,
            **kwargs) -> Dict[str, Union[torch.Tensor, int, float]]:
        """Generate text auto-regressively with stats

        stats include total time, total tokens, and mean token time:
        - total time: total time to generate text
        - total tokens: total number of tokens generated
        - mean token time: mean time to generate a token

        Parameters
        ----------
        input_ids : torch.Tensor
            input token ids
        max_new_tokens : int, optional
            maximum number of tokens to generate, by default 30
        temperature : float, optional
            temperature for sampling tokens from, by default 1.

        Returns
        -------
        Dict[str, Union[torch.Tensor, int, float]]
            stats with generated token ids
        """
        # for calculating stats
        start_time = time.perf_counter()
        init_length = input_ids.shape[1]

        past_key_values = None
        for _ in range(max_new_tokens):
            if self._is_terminator(input_ids[:, -1]):
                break
            if self.use_cache:
                if past_key_values is None:
                    past_key_values = self.target_model(
                        input_ids[:, :-1],
                        use_cache=True,
                    ).past_key_values
                model_inputs = self.target_model.prepare_inputs_for_generation(
                    input_ids,
                    past_key_values=past_key_values,
                )
                outputs = self.target_model(**model_inputs, use_cache=True)
                next_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
            else:
                next_logits = self.target_model(
                    input_ids,
                    use_cache=False,
                ).logits[:, -1, :]
            probs = self._logits2probs(next_logits, temperature=temperature)
            token_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, token_id), dim=1)

        # for calculating stats
        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_tokens = input_ids.shape[1] - init_length
        mean_token_time = total_time / total_tokens

        return {
            "output_ids": input_ids,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "mean_token_time": mean_token_time,
        }


class SpeculativeDecoder(BaseDecoder):
    """Speculative decoding with speculative sampling
    """
    def __init__(
            self,
            draft_model: torch.nn.Module,
            **args) -> None:
        """Initialize the decoder

        Parameters
        ----------
        draft_model : torch.nn.Module
            draft model to generate text
        """
        super().__init__(**args)
        self.draft_model = draft_model
        self.vocab_size = draft_model.config.vocab_size

    def _acceptance_prob(
            self,
            target_prob: torch.Tensor,
            draft_prob: torch.Tensor,
            token_id: torch.Tensor) -> torch.Tensor:
        """Calculate acceptance probability

        Parameters
        ----------
        target_prob : torch.Tensor
            target probability distribution
        draft_prob : torch.Tensor
            draft probability distribution
        token_id : torch.Tensor
            token id

        Returns
        -------
        torch.Tensor
            acceptance probability
        """
        return torch.min(
                torch.tensor([1], device=target_prob.device),
                target_prob[:, token_id] / draft_prob[:, token_id],
            )  # (batch, vocab)

    def _adjusted_probs(
            self,
            target_probs: torch.Tensor,
            draft_probs: torch.Tensor) -> torch.Tensor:
        """Adjust probability distribution

        Parameters
        ----------
        target_probs : torch.Tensor
            target probability distribution
        draft_probs : torch.Tensor
            draft probability distribution

        Returns
        -------
        torch.Tensor
            adjusted probability distribution
        """
        return F.normalize(
                torch.clamp(target_probs[:, :] - draft_probs[:, :], min=0),
                p=1, dim=1,
            )  # (batch, vocab)

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 30,
            temperature: float = 1.,
            n_lookahead: int = 5,
            **kwargs) -> torch.Tensor:
        """Generate text with speculative decoding

        Parameters
        ----------
        input_ids : torch.Tensor
            input token ids
        max_new_tokens : int, optional
            maximum number of tokens to generate, by default 30
        temperature : float, optional
            temperature for sampling tokens from, by default 1.
        n_lookahead : int, optional
            number of tokens to look ahead, by default 5

        Returns
        -------
        torch.Tensor
            generated token ids
        """
        # 1. Initialize
        T = input_ids.shape[1] + max_new_tokens - 1
        # constant value: position to stop generation

        n = input_ids.shape[1] - 1
        # variable: the end position of the confirmed token

        draft_probs = torch.empty((
            input_ids.shape[0],
            n_lookahead,
            self.vocab_size,
        ), device=input_ids.device)
        target_probs = torch.empty((
            input_ids.shape[0],
            n_lookahead + 1,
            self.vocab_size,
        ), device=input_ids.device)
        # to store tentetive probability distributions
        # shape: (batch, n_lookahead, vocab)

        draft_past_key_values = None
        target_past_key_values = None
        # to store past key values for fast generation

        # while not EOS/PAD, and not reach the end position (= T)
        while (n < T and not self._is_terminator(input_ids[:, -1])):

            # 2. Sample draft auto-regressively
            draft_token_ids = input_ids
            # input_ids.shape == draft_token_ids.shape -> (batch, sequence)

            draft_past_key_values = None
            for i in range(n_lookahead):
                if self.use_cache:
                    if draft_past_key_values is None:
                        draft_past_key_values = self.draft_model(
                            input_ids[:, :-1],
                            use_cache=True,
                        ).past_key_values
                    model_inputs = self.draft_model.prepare_inputs_for_generation(
                        input_ids,
                        past_key_values=draft_past_key_values,
                    )
                    outputs = self.draft_model(**model_inputs, use_cache=True)
                    next_logits = outputs.logits[:, -1, :]
                    tmp_draft_past_key_values = outputs.past_key_values
                else:
                    next_logits = self.draft_model(
                        input_ids,
                        use_cache=False,
                    ).logits[:, -1, :]
                draft_probs[:, i, :] = self._logits2probs(
                    next_logits,
                    temperature=temperature,
                )  # logits.shape -> (batch, sequence, vocab)
                token_id = torch.multinomial(
                    draft_probs[:, i, :],
                    num_samples=1,
                )  # sample 1 token from the probability distribution
                draft_token_ids = torch.cat((draft_token_ids, token_id), dim=1)

            # 3. In parallel, compute K + 1 sets of logits from drafts
            if self.use_cache:
                if target_past_key_values is None:
                    target_past_key_values = self.target_model(
                        input_ids[:, :-1],
                        use_cache=True,
                    ).past_key_values
                model_inputs = self.target_model.prepare_inputs_for_generation(
                    input_ids,
                    past_key_values=target_past_key_values,
                )
                outputs = self.target_model(
                    **model_inputs,
                    use_cache=True,
                )  # parallel computation
                logits = outputs.logits[:, -n_lookahead - 1:, :]
                tmp_target_past_key_values = outputs.past_key_values
            else:
                logits = self.target_model(
                    input_ids,
                    use_cache=False,
                ).logits[:, -n_lookahead - 1:, :]  # parallel computation
            target_probs[:, :, :] = self._logits2probs(
                logits,
                temperature=temperature,
            )  # modify all logits to probs at once
            # -n_lookahead - 1, -n_lookahead, ..., -1
            # get 1 extra probs for "8. If all tokens are accepted,
            # sample an extra token from target model" to be processed later
            # target_probs.shape -> (batch, n_lookahead + 1, vocab)

            is_all_tokens_accepted = True
            for i in range(n_lookahead):
                # while not EOS/PAD, and not reach the end position (= T)
                # like the outer while loop, but this is for the inner for loop
                if (not n < T or self._is_terminator(input_ids[:, -1])):
                    is_all_tokens_accepted = False
                    break

                # 4. Sample r  from a uniform distribution
                r = torch.rand(1, device=input_ids.device)

                # 5. if r < min(1, target_model_prob / draft_model_prob)
                # 6. Then accept the draft token
                if r < self._acceptance_prob(
                        target_probs[:, i],
                        draft_probs[:, i],
                        draft_token_ids[:, -n_lookahead + i]):
                    input_ids = torch.cat((
                        input_ids,
                        draft_token_ids[:, -n_lookahead + i].unsqueeze(1),
                    ), dim=1)
                    # unsqueeze method for
                    # "RuntimeError: Tensors must have same number of
                    # dimensions: got 2 and 1"
                    n += 1

                # 7. Otherwise, resample a token from
                # the adjusted distribution, and exit the loop
                else:
                    probs = self._adjusted_probs(
                        target_probs[:, i, :],
                        draft_probs[:, i, :],
                    )
                    token_id = torch.multinomial(probs, num_samples=1)
                    # sample 1 token from the "adjusted" probability
                    input_ids = torch.cat((input_ids, token_id), dim=1)
                    is_all_tokens_accepted = False
                    break

            # 8. If all tokens are accepted,
            # sample an extra token from target model
            if is_all_tokens_accepted and n < T:
                token_id = torch.multinomial(
                    target_probs[:, -1, :],
                    num_samples=1,
                )
                # Use pre-acquired distributions in
                # "3. In parallel, compute K + 1 sets of logits from drafts"
                input_ids = torch.cat((input_ids, token_id), dim=1)
                if self.use_cache:
                    target_past_key_values = tmp_target_past_key_values
                    draft_past_key_values = tmp_draft_past_key_values
            n += 1  # count for rejection, extra pick
        return input_ids

    @torch.no_grad()
    def generate_with_stats(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 30,
            temperature: float = 1.,
            n_lookahead: int = 5,
            **kwargs) -> Dict[str, Union[torch.Tensor, int, float, List[Union[str, torch.Tensor, None]]]]:
        """Generate text with speculative decoding with stats

        stats include:
        - total time: total time to generate text
        - total tokens: total number of tokens generated
        - mean token time: mean time to generate a token
        - acceptance rate: acceptance rate of tokens
        - accepted count: number of accepted tokens
        - rejected count: number of rejected tokens
        - extra count: number of extra tokens
        - logs: logs for each token generation step

        each log is a list of 3 elements:
        - ["S", None, None]: start of the loop
        - ["A", accepted_token_id, None]: draft token is accepted
        - ["R", rejected_token_id, resampled_token_id]:
            draft token is rejected and resampled
            next log is must be ["S", None, None]
        - ["E", extra_token_id, None]:
            extra token is sampled because all tokens are accepted
            next log is must be ["S", None, None]

        Parameters
        ----------
        input_ids : torch.Tensor
            input token ids
        max_new_tokens : int, optional
            maximum number of tokens to generate, by default 30
        temperature : float, optional
            temperature for sampling tokens from, by default 1.
        n_lookahead : int, optional
            number of tokens to look ahead, by default 5

        Returns
        -------
        Dict[str, Union[torch.Tensor, int, float, List[Union[str, torch.Tensor, None]]]]
            stats with generated token ids
        """

        # for calculating stats
        start_time = time.perf_counter()
        init_length = input_ids.shape[1]
        accepted_count = 0
        rejected_count = 0
        extra_count = 0
        logs = []
        # ["S", None, None] -> start of the loop
        # ["A", accepted_token_id, None] -> accepted
        # ["R", rejected_token_id, resampled_token_id] -> rejected
        # ["E", extra_token_id, None] -> extra

        T = input_ids.shape[1] + max_new_tokens - 1
        n = input_ids.shape[1] - 1
        draft_past_key_values = None
        target_past_key_values = None
        draft_probs = torch.empty((
            input_ids.shape[0],
            n_lookahead,
            self.vocab_size
        ), device=input_ids.device)
        target_probs = torch.empty((
            input_ids.shape[0],
            n_lookahead + 1,
            self.vocab_size,
        ), device=input_ids.device)

        while (n < T and not self._is_terminator(input_ids[:, -1])):
            logs.append(["S", None, None])  # logs: ["S", None, None]
            draft_token_ids = input_ids
            for i in range(n_lookahead):
                if self.use_cache:
                    if draft_past_key_values is None:
                        draft_past_key_values = self.draft_model(
                            input_ids[:, :-1],
                            use_cache=True,
                        ).past_key_values
                    model_inputs = self.draft_model.prepare_inputs_for_generation(
                        input_ids,
                        past_key_values=draft_past_key_values,
                    )
                    outputs = self.draft_model(**model_inputs, use_cache=True)
                    next_logits = outputs.logits[:, -1, :]
                    tmp_draft_past_key_values = outputs.past_key_values
                else:
                    next_logits = self.draft_model(
                        input_ids,
                        use_cache=False,
                    ).logits[:, -1, :]
                draft_probs[:, i, :] = self._logits2probs(
                    next_logits,
                    temperature=temperature,
                )
                token_id = torch.multinomial(
                    draft_probs[:, i, :],
                    num_samples=1,
                )
                draft_token_ids = torch.cat((draft_token_ids, token_id), dim=1)

            if self.use_cache:
                if target_past_key_values is None:
                    target_past_key_values = self.target_model(
                        input_ids[:, :-1],
                        use_cache=True,
                    ).past_key_values
                model_inputs = self.target_model.prepare_inputs_for_generation(
                    draft_token_ids,
                    past_key_values=target_past_key_values,
                )
                outputs = self.target_model(**model_inputs, use_cache=True)
                logits = outputs.logits[:, -n_lookahead - 1:, :]
                tmp_target_past_key_values = outputs.past_key_values
            else:
                logits = self.target_model(
                    draft_token_ids,
                    use_cache=False,
                ).logits[:, -n_lookahead - 1:, :]
            target_probs[:, :, :] = self._logits2probs(
                logits,
                temperature=temperature,
            )

            is_all_tokens_accepted = True
            for i in range(n_lookahead):
                if (not n < T or self._is_terminator(input_ids[:, -1])):
                    is_all_tokens_accepted = False
                    break
                r = torch.rand(1, device=input_ids.device)
                if r < self._acceptance_prob(
                        target_probs[:, i],
                        draft_probs[:, i],
                        draft_token_ids[:, -n_lookahead + i]):
                    input_ids = torch.cat((
                        input_ids,
                        draft_token_ids[:, -n_lookahead + i].unsqueeze(1),
                    ), dim=1)
                    n += 1
                    accepted_count += 1  # for calculating stats
                    logs.append([
                        "A",
                        draft_token_ids[:, -n_lookahead + i].unsqueeze(1),
                        None,
                    ])  # logs: ["A", accepted_token_id, None]
                else:
                    probs = self._adjusted_probs(
                        target_probs[:, i, :],
                        draft_probs[:, i, :],
                    )
                    token_id = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat((input_ids, token_id), dim=1)
                    is_all_tokens_accepted = False
                    rejected_count += 1  # for calculating stats
                    logs.append([
                        "R",
                        draft_token_ids[:, -n_lookahead + i].unsqueeze(1),
                        token_id,
                    ])  # logs: ["R", rejected_token_id, resampled_token_id]
                    break
            if is_all_tokens_accepted and n < T:
                token_id = torch.multinomial(
                    target_probs[:, -1, :],
                    num_samples=1,
                )
                input_ids = torch.cat((input_ids, token_id), dim=1)
                if self.use_cache:
                    target_past_key_values = tmp_target_past_key_values
                    draft_past_key_values = tmp_draft_past_key_values
                extra_count += 1  # for calculating stats
                logs.append(["E", token_id, None])
            n += 1

        # for calculating stats
        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_tokens = input_ids.shape[1] - init_length
        mean_token_time = total_time / total_tokens
        acceptance_rate = accepted_count / (accepted_count + rejected_count)

        return {
            "output_ids": input_ids,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "mean_token_time": mean_token_time,
            "acceptance_rate": acceptance_rate,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "extra_count": extra_count,
            "logs": logs,
        }
