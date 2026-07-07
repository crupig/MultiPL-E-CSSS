"""
This script produces completions for roughly any AutoModelForCausalLM.
"""
from multipl_e.completions_with_logprobs import make_main, stop_at_stop_token, partial_arg_parser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
from typing import List
import torch.nn.functional as F


class Model:
    def __init__(self, name, revision, model_kwargs, tokenizer_name=None, tokenizer_revision=None):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype=dtype, trust_remote_code=True, **model_kwargs
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or name,
            revision=tokenizer_revision or revision,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        assert (
            self.tokenizer.pad_token is not None
        ), "tokenizer has neither pad_token nor eos_token"

        self._all_special_token_ids = self.tokenizer.all_special_ids

        assert (
            len(self._all_special_token_ids) >= 1
        ), "tokenizer.all_special_ids() is empty"
        assert (
            self.tokenizer.pad_token_id in self._all_special_token_ids
        ), "pad_token_id not in all_special_ids"
        assert (
            self.tokenizer.eos_token_id in self._all_special_token_ids
        ), "eos_token_id not in all_special_ids"

    # def completion_tensors_batch1(
    #     self,
    #     prompts: list,
    #     max_length: int,
    #     temperature: float,
    #     top_p: float,
    # ):
    #     self.model.eval() # Not essential, but just in case.

    #     prompt = prompts[0]

    #     inputs_ids = self.tokenizer(
    #         prompt,
    #         padding=False,
    #         return_tensors="pt",
    #         return_token_type_ids=False,
    #         truncation=True,
    #         max_length=max_length - 1,
    #     ).to("cuda")["input_ids"]


    #     with torch.no_grad():
    #         output_ids = self.model.generate(
    #             inputs_ids,
    #             do_sample=True,
    #             use_cache=True,
    #             top_p=top_p,
    #             temperature=temperature,
    #             max_length=max_length,
    #             pad_token_id=self.tokenizer.pad_token_id,
    #             output_scores=True,
    #             return_dict_in_generate=True,
    #         )

    #     generated_ids = output_ids.sequences[0][inputs_ids.shape[-1]:]
    #     probabilities = []
    #     # Compute log probabilities
    #     for iscore, score in enumerate(output_ids.scores):  # one score per generated token
    #         token_id = generated_ids[iscore]
    #         prob = F.log_softmax(score, dim=-1)
    #         prob = prob[0, token_id].item()
    #         probabilities.append(prob)
        
    #     output = [output_ids.sequences[0]]
    #     return output, probabilities

    
    def completion_tensors(
                            self,
                            prompts: list,
                            max_length: int,
                            temperature: float,
                            top_p: float,
                        ):
        self.model.eval()

        # Tokenize all prompts at once
        input_batch = self.tokenizer(
            prompts,
            padding=True,  # Pad to match max length in batch
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True,
            max_length=max_length - 1,
        ).to("cuda")

        input_ids = input_batch["input_ids"]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                do_sample=True,
                use_cache=True,
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        
        all_outputs = []
        all_logprobs = []

        # output_ids.scores is a list of length T (num generated tokens),
        # each element is [batch_size, vocab_size]
        # So we need to process logprobs per example in the batch
        generated = output_ids.sequences  # [batch_size, total_length]
        scores = output_ids.scores       # List[T] of tensors [batch_size, vocab_size]
        
        for batch_idx in range(len(prompts)):
            input_len = (input_ids[batch_idx] != pad_id).sum().item()
            gen_ids = generated[batch_idx][input_len:]
            logprobs = []
            for t, score_t in enumerate(scores):
                token_id = gen_ids[t]
                prob = F.log_softmax(score_t[batch_idx], dim=-1)
                logprobs.append(prob[token_id].item())

            all_outputs.append(generated[batch_idx])
            all_logprobs.append(logprobs)

        return all_outputs, all_logprobs

    def _is_normal_token_id(self, token_id: int) -> bool:
        return token_id not in self._all_special_token_ids

    def _is_pad_or_bos_token_id(self, token_id: int) -> bool:
        if token_id == self.tokenizer.pad_token_id:
            return True
        if self.tokenizer.bos_token_id is not None and token_id == self.tokenizer.bos_token_id:
            return True
        return False

    def _remove_padding_and_stop_at_special_tokens(self, token_id_list: List[int]):
        pad_token_id = self.tokenizer.pad_token_id
        # bos_token_id may be None
        bos_token_id = self.tokenizer.bos_token_id
        # Removes all the pad tokens or BOS tokens on the left-hand side using the 
        # pad token ID. This is more robust than looking for the string representation of
        # the pad token. Thus the prompt can begin with the literal string
        # "<|endoftext|>" (which is a common representation of the pad token).
        left_padding_removed = itertools.dropwhile(
            self._is_pad_or_bos_token_id, token_id_list
        )
        # Returns all tokens to the left of the first special token. This has
        # the effect of removing all right-hand padding. Moreover, it also
        # stops generation at other special tokens. For example, consider
        # StarCoder 2, where a completion may reach the end of a file and then
        # continue onto a second file: A<file_sep>B. The code below removes
        # <file_sep>B and only produces A.
        right_specials_removed = itertools.takewhile(
            self._is_normal_token_id, left_padding_removed
        )
        return list(right_specials_removed)

    def decode_single_output(self, output_tensor, prompt):
        output_token_ids = self._remove_padding_and_stop_at_special_tokens(
            output_tensor.tolist()
        )
        detok_hypo_str = self.tokenizer.decode(
            output_token_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        # Skip the prompt (which may even have stop_tokens)
        return detok_hypo_str[len(prompt) :]

    def completions(
        self, prompts: str, max_tokens: int, temperature: float, top_p, stop
    ):
        prompts = [prompt.strip() for prompt in prompts]
        # output_tensors = self.completion_tensors(
        output_tensors, probs = self.completion_tensors(
            prompts,
            max_tokens,
            temperature,
            top_p,
        )
        
        # post process output and log probabilities
        outputs, log_probabilities = [], []
        for (prompt, output_tensor, prob) in zip(prompts, output_tensors, probs):
            output, lp = stop_at_stop_token(
                self.decode_single_output(output_tensor, prompt),
                stop, prob, self.tokenizer
            )
            outputs.append(output)
            log_probabilities.append(lp)

        return outputs, log_probabilities


def automodel_partial_arg_parser():
    """
    This is also used by peftmodel.py.
    """
    args = partial_arg_parser()
    args.add_argument("--name", type=str, required=True)
    args.add_argument("--revision", type=str)
    args.add_argument("--tokenizer_name", type=str)
    args.add_argument("--tokenizer_revision", type=str)
    args.add_argument("--name-override", type=str)
    args.add_argument("--flash-attention2", action="store_true")
    return args


def do_name_override(args):
    """
    Applies the --name-override flag, or uses the model name, correcting / and - which the rest of
    the toolchain does not like.
    """
    if args.name_override:
        name = args.name_override
    else:
        name = args.name.replace("/", "_").replace("-", "_")
    return name


def main():
    args = automodel_partial_arg_parser()
    args = args.parse_args()
    model_kwargs = { }
    if args.flash_attention2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = Model(
        args.name, args.revision,
        model_kwargs=model_kwargs,
        tokenizer_name=args.tokenizer_name,
        tokenizer_revision=args.tokenizer_revision,
    )
    name = do_name_override(args)
    make_main(args, name, model.completions)


if __name__ == "__main__":
    main()
