from typing import Dict, List, Union
import time

import torch
import torch.nn.functional as F


class BaseDecoder:
    """_summary_
    """
    def __init__(
            self,
            target_model: torch.nn.Module,
            eos_token_id: int = None,
            pad_token_id: int = None,
            use_cache: bool = False,
            **kwargs) -> None:
        self.target_model = target_model
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_cache = use_cache

    def _is_terminator(self, token_id: torch.Tensor) -> bool:
        if ((self.eos_token_id is not None and token_id == self.eos_token_id) or
                (self.pad_token_id is not None and token_id == self.pad_token_id)):
            # terminator: EOS or PAD
            # "terminator is specified" and "token_id is equal to the terminator"
            return True
        else:
            return False

    def _logits2probs(
            self,
            logits: torch.Tensor,
            temperature: float = 1.) -> torch.Tensor:
        # TODO: top_k, top_p filtering
        logits = logits / (temperature + torch.finfo(logits.dtype).eps)  # temperature adjustment
        probs = F.softmax(logits, dim=-1)  # get probability distribution (sum of probs is equal to 1)
        return probs

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            **kwargs) -> torch.Tensor:
        pass

    @torch.no_grad()
    def generate_with_stats(self,
            input_ids: torch.Tensor,
            **kwargs) -> torch.Tensor:
        pass


class AutoRegressiveDecoder(BaseDecoder):
    """_summary_

    Parameters
    ----------
    BaseDecoder : _type_
        _description_
    """
    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 30,
            temperature: float = 1.,
            **kwargs) -> torch.Tensor:
        """_summary_

        Parameters
        ----------
        input_ids : torch.Tensor
            _description_
        max_new_tokens : int, optional
            _description_, by default 30
        temperature : float, optional
            _description_, by default 1.

        Returns
        -------
        torch.Tensor
            _description_
        """
        past_key_values = None
        for _ in range(max_new_tokens):
            if self._is_terminator(input_ids[:, -1]):
                break
            if self.use_cache:
                if past_key_values is None:
                    past_key_values = self.target_model(input_ids[:, :-1], use_cache=True).past_key_values
                model_inputs = self.target_model.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
                outputs = self.target_model(**model_inputs, use_cache=True)
                next_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
            else:
                next_logits = self.target_model(input_ids, use_cache=False).logits[:, -1, :]
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
            **args) -> torch.Tensor:
        # for calculating stats
        start_time = time.perf_counter()
        init_length = input_ids.shape[1]

        past_key_values = None
        for _ in range(max_new_tokens):
            if self._is_terminator(input_ids[:, -1]):
                break
            if self.use_cache:
                if past_key_values is None:
                    past_key_values = self.target_model(input_ids[:, :-1], use_cache=True).past_key_values
                model_inputs = self.target_model.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
                outputs = self.target_model(**model_inputs, use_cache=True)
                next_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
            else:
                next_logits = self.target_model(input_ids, use_cache=False).logits[:, -1, :]
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
    """_summary_

    Parameters
    ----------
    BaseDecoder : _type_
        _description_
    """
    def __init__(self,
            draft_model: torch.nn.Module,
            **args) -> None:
        super().__init__(**args)
        self.draft_model = draft_model
        self.vocab_size = draft_model.config.vocab_size

    def _acceptance_prob(
            self,
            target_prob: torch.Tensor,
            draft_prob: torch.Tensor,
            token_id: torch.Tensor) -> torch.Tensor:
        return torch.min(
                torch.tensor([1], device=target_prob.device),
                target_prob[:, token_id] / draft_prob[:, token_id],
            )  # (batch, vocab)

    def _adjusted_probs(
            self,
            target_probs: torch.Tensor,
            draft_probs: torch.Tensor) -> torch.Tensor:
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
        """_summary_

        Parameters
        ----------
        input_ids : torch.Tensor
            _description_
        max_new_tokens : int, optional
            _description_, by default 30
        temperature : float, optional
            _description_, by default 1.
        n_lookahead : int, optional
            _description_, by default 5

        Returns
        -------
        torch.Tensor
            _description_
        """

        # 1. Initialize
        T = input_ids.shape[1] + max_new_tokens - 1  # constant value: position to stop generation
        n = input_ids.shape[1] - 1  # variable: the end position of the confirmed token
        draft_probs = torch.empty((input_ids.shape[0], n_lookahead, self.vocab_size), device=input_ids.device)
        target_probs = torch.empty((input_ids.shape[0], n_lookahead + 1, self.vocab_size), device=input_ids.device)
        # to store tentetive probability distributions; (batch, n_lookahead, vocab)

        while (n < T and not self._is_terminator(input_ids[:, -1])):  # while not EOS/PAD, and not reach the end position (= T)

            # 2. Sample draft auto-regressively
            draft_token_ids = input_ids  # input_iss.shape == draft_token_ids.shape -> (batch, sequence)
            draft_past_key_values = None
            for i in range(n_lookahead):
                if self.use_cache:
                    if draft_past_key_values is None:
                        draft_past_key_values = self.draft_model(input_ids[:, :-1], use_cache=True).past_key_values
                    model_inputs = self.draft_model.prepare_inputs_for_generation(input_ids, past_key_values=draft_past_key_values)
                    outputs = self.draft_model(**model_inputs, use_cache=True)
                    next_logits = outputs.logits[:, -1, :]
                    tmp_draft_past_key_values = outputs.past_key_values
                else:
                    next_logits = self.draft_model(input_ids, use_cache=False).logits[:, -1, :]
                draft_probs[:, i, :] = self._logits2probs(next_logits, temperature=temperature)  # logits.shape -> (batch, sequence, vocab)
                token_id = torch.multinomial(draft_probs[:, i, :], num_samples=1)  # sample 1 token from the probability distribution
                draft_token_ids = torch.cat((draft_token_ids, token_id), dim=1)

            # 3. In parallel, compute K + 1 sets of logits from drafts
            if self.use_cache:
                if target_past_key_values is None:
                    target_past_key_values = self.target_model(input_ids[:, :-1], use_cache=True).past_key_values
                model_inputs = self.target_model.prepare_inputs_for_generation(input_ids, past_key_values=target_past_key_values)
                outputs = self.target_model(**model_inputs, use_cache=True)  # parallel computation
                logits = outputs.logits
                tmp_target_past_key_values = outputs.past_key_values
            else:
                logits = self.target_model(input_ids, use_cache=False).logits  # parallel computation
            for i in range(n_lookahead + 1):
                target_probs[:, i, :] = self._logits2probs(logits[:, -n_lookahead - 1 + i, :], temperature=temperature)  # -n_lookahead - 1, -n_lookahead, ..., -1
            # get 1 extra probs for "8. If all tokens are accepted, sample an extra token from target model" to be processed later
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
                if r < self._acceptance_prob(target_probs[:, i], draft_probs[:, i], draft_token_ids[:, -n_lookahead + i]):
                    input_ids = torch.cat((input_ids, draft_token_ids[:, -n_lookahead + i].unsqueeze(1)), dim=1)
                    # unsqueeze method for "RuntimeError: Tensors must have same number of dimensions: got 2 and 1"
                    n += 1

                # 7. Otherwise, resample a token from the adjusted distribution, and exit the loop
                else:
                    probs = self._adjusted_probs(target_probs[:, i, :], draft_probs[:, i, :])
                    token_id = torch.multinomial(probs, num_samples=1)  # sample 1 token from the "adjusted" probability distribution
                    input_ids = torch.cat((input_ids, token_id), dim=1)
                    is_all_tokens_accepted = False
                    break

            # 8. If all tokens are accepted, sample an extra token from target model
            if is_all_tokens_accepted and n < T:
                token_id = torch.multinomial(target_probs[:, -1, :], num_samples=1)
                # Use pre-acquired distributions in "3. In parallel, compute K + 1 sets of logits from drafts"
                input_ids = torch.cat((input_ids, token_id), dim=1)
                n += 1
                if self.use_cache:
                    target_past_key_values = tmp_target_past_key_values
                    draft_past_key_values = tmp_draft_past_key_values
        return input_ids

    @torch.no_grad()
    def generate_with_stats(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 30,
            temperature: float = 1.,
            n_lookahead: int = 5,
            **kwargs) -> Dict[str, Union[torch.Tensor, int, List[Union[str, torch.Tensor, torch.Tensor]]]]:

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
        draft_probs = torch.empty((input_ids.shape[0], n_lookahead, self.vocab_size), device=input_ids.device)
        target_probs = torch.empty((input_ids.shape[0], n_lookahead + 1, self.vocab_size), device=input_ids.device)

        while (n < T and not self._is_terminator(input_ids[:, -1])):
            logs.append(["S", None, None])  # logs: ["S", None, None]
            draft_token_ids = input_ids
            for i in range(n_lookahead):
                if self.use_cache:
                    if draft_past_key_values is None:
                        draft_past_key_values = self.draft_model(input_ids[:, :-1], use_cache=True).past_key_values
                    model_inputs = self.draft_model.prepare_inputs_for_generation(input_ids, past_key_values=draft_past_key_values)
                    outputs = self.draft_model(**model_inputs, use_cache=True)
                    next_logits = outputs.logits[:, -1, :]
                    tmp_draft_past_key_values = outputs.past_key_values
                else:
                    next_logits = self.draft_model(input_ids, use_cache=False).logits[:, -1, :]
                draft_probs[:, i, :] = self._logits2probs(next_logits, temperature=temperature)
                token_id = torch.multinomial(draft_probs[:, i, :], num_samples=1)
                draft_token_ids = torch.cat((draft_token_ids, token_id), dim=1)

            if self.use_cache:
                if target_past_key_values is None:
                    target_past_key_values = self.target_model(input_ids[:, :-1], use_cache=True).past_key_values
                model_inputs = self.target_model.prepare_inputs_for_generation(draft_token_ids, past_key_values=target_past_key_values)
                outputs = self.target_model(**model_inputs, use_cache=True)
                logits = outputs.logits
                tmp_target_past_key_values = outputs.past_key_values
            else:
                logits = self.target_model(input_ids, use_cache=False).logits
            for i in range(n_lookahead + 1):
                target_probs[:, i, :] = self._logits2probs(logits[:, -n_lookahead - 1 + i, :], temperature=temperature)

            is_all_tokens_accepted = True
            for i in range(n_lookahead):
                if (not n < T or self._is_terminator(input_ids[:, -1])):
                    is_all_tokens_accepted = False
                    break
                r = torch.rand(1, device=input_ids.device)
                if r < self._acceptance_prob(target_probs[:, i], draft_probs[:, i], draft_token_ids[:, -n_lookahead + i]):
                    input_ids = torch.cat((input_ids, draft_token_ids[:, -n_lookahead + i].unsqueeze(1)), dim=1)
                    n += 1
                    accepted_count += 1  # for calculating stats
                    logs.append(["A", draft_token_ids[:, -n_lookahead + i].unsqueeze(1), None])  # logs: ["A", accepted_token_id, None]
                else:
                    probs = self._adjusted_probs(target_probs[:, i, :], draft_probs[:, i, :])
                    token_id = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat((input_ids, token_id), dim=1)
                    is_all_tokens_accepted = False
                    rejected_count += 1  # for calculating stats
                    logs.append(["R", draft_token_ids[:, -n_lookahead + i].unsqueeze(1), token_id])  # logs: ["R", rejected_token_id, resampled_token_id]
                    break
            if is_all_tokens_accepted and n < T:
                token_id = torch.multinomial(target_probs[:, -1, :], num_samples=1)
                input_ids = torch.cat((input_ids, token_id), dim=1)
                n += 1
                if self.use_cache:
                    target_past_key_values = tmp_target_past_key_values
                    draft_past_key_values = tmp_draft_past_key_values
                extra_count += 1  # for calculating stats
                logs.append(["E", token_id, None])

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