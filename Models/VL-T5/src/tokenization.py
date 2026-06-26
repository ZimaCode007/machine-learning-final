from transformers import T5Tokenizer, T5TokenizerFast
from typing import List


class VLT5TokenizerFast(T5TokenizerFast):
    """T5TokenizerFast extended with <vis_extra_id_N> visual tokens."""

    def __init__(self, *args, vis_extra_ids=100, **kwargs):
        super().__init__(*args, **kwargs)
        self._vis_extra_ids = vis_extra_ids
        vis_tokens = [f"<vis_extra_id_{i}>" for i in range(vis_extra_ids)]
        self.add_tokens(vis_tokens, special_tokens=True)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, vis_extra_ids=100, **kwargs):
        tokenizer = T5TokenizerFast.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        tokenizer.__class__ = cls
        tokenizer._vis_extra_ids = vis_extra_ids
        vis_tokens = [f"<vis_extra_id_{i}>" for i in range(vis_extra_ids)]
        tokenizer.add_tokens(vis_tokens, special_tokens=True)
        return tokenizer
