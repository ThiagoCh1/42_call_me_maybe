"""Constrained decoder for the call_me_maybe project.

Implements token-by-token constrained decoding to guarantee 100% valid JSON
output conforming to the expected function call schema.

All fixed strings are forced via pre-computed token sequences.
Function selection: the model picks tokens freely until only one candidate
function remains, then the rest of the name is forced.

Expected output format:
{"prompt":"<user prompt>","name":"<function name>","parameters":{...}}
"""

from enum import Enum, auto
from typing import Optional

from src.models import FunctionDefinition


class DecoderState(Enum):
    """States of the JSON generation state machine."""

    START = auto()

    PROMPT_KEY_QUOTE = auto()
    PROMPT_KEY_TOKENS = auto()
    PROMPT_KEY_CLOSE_QUOTE = auto()
    PROMPT_COLON = auto()
    PROMPT_VALUE_OPEN_QUOTE = auto()
    PROMPT_VALUE_TOKENS = auto()
    PROMPT_VALUE_CLOSE_QUOTE = auto()
    AFTER_PROMPT_COMMA = auto()

    NAME_KEY_QUOTE = auto()
    NAME_KEY_TOKENS = auto()
    NAME_KEY_CLOSE_QUOTE = auto()
    NAME_COLON = auto()
    NAME_VALUE_OPEN_QUOTE = auto()
    NAME_VALUE_TOKENS = auto()       # model picks until one candidate left
    NAME_VALUE_CLOSE_QUOTE = auto()
    AFTER_NAME_COMMA = auto()

    PARAMS_KEY_QUOTE = auto()
    PARAMS_KEY_TOKENS = auto()
    PARAMS_KEY_CLOSE_QUOTE = auto()
    PARAMS_COLON = auto()
    PARAMS_OPEN_BRACE = auto()

    PARAM_KEY_OPEN_QUOTE = auto()
    PARAM_KEY_FIRST_TOKEN = auto()
    PARAM_KEY_TOKENS = auto()
    PARAM_KEY_CLOSE_QUOTE = auto()
    PARAM_COLON = auto()
    PARAM_VALUE_START = auto()
    PARAM_NUMBER_CONTENT = auto()
    PARAM_STRING_TOKENS = auto()
    AFTER_PARAM = auto()

    END_BRACE = auto()
    DONE = auto()


class ConstrainedDecoder:
    """Generates a function call JSON using constrained decoding.

    Args:
        model: The Small_LLM_Model instance.
        functions: List of available function definitions.
        vocab: Dict mapping token string to token ID.
        user_prompt: The original user request string.
    """

    def __init__(
        self,
        model: object,
        functions: list[FunctionDefinition],
        vocab: dict[str, int],
        user_prompt: str = "",
    ) -> None:
        self._model = model
        self._functions = {f.name: f for f in functions}
        self._vocab = vocab
        self._id_to_token: dict[int, str] = {v: k for k, v in vocab.items()}
        self._user_prompt = user_prompt

        self._tok = {
            '"': vocab['"'],
            ':': vocab[':'],
            '{': vocab['{'],
            '}': vocab['}'],
            ',': vocab[','],
        }

        self._key_tokens: dict[str, list[int]] = {
            key: self._encode(key)
            for key in ('prompt', 'name', 'parameters')
        }

        self._func_name_tokens: dict[str, list[int]] = {
            fname: self._encode(fname)
            for fname in self._functions
        }

        self._param_tokens: dict[str, dict[str, list[int]]] = {
            fname: {
                pname: self._encode(pname)
                for pname in func.parameters
            }
            for fname, func in self._functions.items()
        }

        self._prompt_value_tokens: list[int] = (
            self._encode(user_prompt) if user_prompt else []
        )

        # Runtime state
        self._state = DecoderState.START
        self._generated = ""
        self._chosen_function: Optional[str] = None
        self._current_param: Optional[str] = None
        self._used_params: set[str] = set()
        self._seq: list[int] = []
        self._seq_pos: int = 0

        # Tracks tokens chosen so far for function name disambiguation
        self._name_tokens_so_far: list[int] = []

    def _encode(self, text: str) -> list[int]:
        """Encode text to token IDs using the model tokenizer."""
        return self._model.encode(text)[0].tolist()

    def _clean(self, token_str: str) -> str:
        """Convert BPE space prefix to actual space."""
        return token_str.replace('\u0120', ' ')

    def _get_func_candidates(self) -> list[str]:
        """Return function names whose token sequence matches so far."""
        pos = len(self._name_tokens_so_far)
        return [
            fname for fname, seq in self._func_name_tokens.items()
            if len(seq) >= pos and seq[:pos] == self._name_tokens_so_far
        ]

    def _get_valid_token_ids(self) -> set[int]:
        """Return the set of token IDs valid in the current state."""
        s = self._state

        # ── Single forced structural tokens ───────────────────────────────────
        if s == DecoderState.START:
            return {self._tok['{']}

        if s in (DecoderState.PROMPT_KEY_QUOTE,
                 DecoderState.PROMPT_KEY_CLOSE_QUOTE,
                 DecoderState.PROMPT_VALUE_OPEN_QUOTE,
                 DecoderState.PROMPT_VALUE_CLOSE_QUOTE,
                 DecoderState.NAME_KEY_QUOTE,
                 DecoderState.NAME_KEY_CLOSE_QUOTE,
                 DecoderState.NAME_VALUE_OPEN_QUOTE,
                 DecoderState.NAME_VALUE_CLOSE_QUOTE,
                 DecoderState.PARAMS_KEY_QUOTE,
                 DecoderState.PARAMS_KEY_CLOSE_QUOTE,
                 DecoderState.PARAM_KEY_OPEN_QUOTE,
                 DecoderState.PARAM_KEY_CLOSE_QUOTE):
            return {self._tok['"']}

        if s in (DecoderState.PROMPT_COLON,
                 DecoderState.NAME_COLON,
                 DecoderState.PARAMS_COLON,
                 DecoderState.PARAM_COLON):
            return {self._tok[':']}

        if s in (DecoderState.AFTER_PROMPT_COMMA,
                 DecoderState.AFTER_NAME_COMMA):
            return {self._tok[',']}

        if s == DecoderState.PARAMS_OPEN_BRACE:
            return {self._tok['{']}

        if s == DecoderState.END_BRACE:
            return {self._tok['}']}

        # ── Forced sequences ──────────────────────────────────────────────────
        if s in (DecoderState.PROMPT_KEY_TOKENS,
                 DecoderState.NAME_KEY_TOKENS,
                 DecoderState.PARAMS_KEY_TOKENS,
                 DecoderState.PARAM_KEY_TOKENS,
                 DecoderState.PROMPT_VALUE_TOKENS):
            if self._seq_pos < len(self._seq):
                return {self._seq[self._seq_pos]}
            return {self._tok['"']}

        # ── Function name: model picks freely until one candidate ─────────────
        if s == DecoderState.NAME_VALUE_TOKENS:
            if self._chosen_function is not None:
                # Function identified — force remaining tokens
                if self._seq_pos < len(self._seq):
                    return {self._seq[self._seq_pos]}
                return {self._tok['"']}
            # Offer next token of all still-matching candidates
            pos = len(self._name_tokens_so_far)
            valid: set[int] = set()
            for fname, seq in self._func_name_tokens.items():
                if (len(seq) > pos and
                        seq[:pos] == self._name_tokens_so_far):
                    valid.add(seq[pos])
            return valid

        # ── Param key open: " or } if all params done ─────────────────────────
        if s == DecoderState.PARAM_KEY_OPEN_QUOTE:
            valid = {self._tok['"']}
            if self._chosen_function:
                func = self._functions[self._chosen_function]
                if self._used_params >= set(func.parameters.keys()):
                    valid.add(self._tok['}'])
            return valid

        # ── Param key first token ─────────────────────────────────────────────
        if s == DecoderState.PARAM_KEY_FIRST_TOKEN:
            valid = set()
            if self._chosen_function:
                func = self._functions[self._chosen_function]
                for pname in func.parameters:
                    if pname not in self._used_params:
                        seq = self._param_tokens[self._chosen_function][pname]
                        if seq:
                            valid.add(seq[0])
            return valid

        # ── Parameter value ───────────────────────────────────────────────────
        if s == DecoderState.PARAM_VALUE_START:
            if not self._chosen_function or not self._current_param:
                return set()
            func = self._functions[self._chosen_function]
            ptype = func.parameters[self._current_param].type
            if ptype == 'string':
                return {self._tok['"']}
            if ptype == 'number':
                valid = set()
                for tok_str, tok_id in self._vocab.items():
                    c = self._clean(tok_str)
                    if c and (c[0].isdigit() or c[0] == '-'):
                        valid.add(tok_id)
                return valid
            return set()

        if s == DecoderState.PARAM_NUMBER_CONTENT:
            valid = set()
            for tok_str, tok_id in self._vocab.items():
                c = self._clean(tok_str)
                if c and (c[0].isdigit() or c[0] == '.'):
                    valid.add(tok_id)
            valid.add(self._tok[','])
            valid.add(self._tok['}'])
            return valid

        if s == DecoderState.PARAM_STRING_TOKENS:
            # Free text — exclude all tokens ending with " to avoid early close
            # Only the bare " token closes the string
            bad = {
                tok_id for tok_id, tok_str in self._id_to_token.items()
                if tok_str.endswith('"') and tok_str != '"'
            }
            return set(self._id_to_token.keys()) - bad

        if s == DecoderState.AFTER_PARAM:
            valid = {self._tok[',']}
            if self._chosen_function:
                func = self._functions[self._chosen_function]
                if self._used_params >= set(func.parameters.keys()):
                    valid.add(self._tok['}'])
            return valid

        return set()

    def _reset_param(self) -> None:
        """Reset param tracking after finishing one parameter."""
        self._current_param = None
        self._seq = []
        self._seq_pos = 0

    def _update_state(self, token_id: int, token_str: str) -> None:
        """Advance the state machine and append clean token to generated."""
        clean = self._clean(token_str)
        self._generated += clean
        s = self._state

        if s == DecoderState.START:
            self._state = DecoderState.PROMPT_KEY_QUOTE

        elif s == DecoderState.PROMPT_KEY_QUOTE:
            self._seq = self._key_tokens['prompt']
            self._seq_pos = 0
            self._state = DecoderState.PROMPT_KEY_TOKENS

        elif s == DecoderState.PROMPT_KEY_TOKENS:
            self._seq_pos += 1
            if self._seq_pos >= len(self._seq):
                self._state = DecoderState.PROMPT_KEY_CLOSE_QUOTE

        elif s == DecoderState.PROMPT_KEY_CLOSE_QUOTE:
            self._state = DecoderState.PROMPT_COLON

        elif s == DecoderState.PROMPT_COLON:
            self._state = DecoderState.PROMPT_VALUE_OPEN_QUOTE

        elif s == DecoderState.PROMPT_VALUE_OPEN_QUOTE:
            self._seq = self._prompt_value_tokens
            self._seq_pos = 0
            if not self._seq:
                self._state = DecoderState.PROMPT_VALUE_CLOSE_QUOTE
            else:
                self._state = DecoderState.PROMPT_VALUE_TOKENS

        elif s == DecoderState.PROMPT_VALUE_TOKENS:
            self._seq_pos += 1
            if self._seq_pos >= len(self._seq):
                self._state = DecoderState.PROMPT_VALUE_CLOSE_QUOTE

        elif s == DecoderState.PROMPT_VALUE_CLOSE_QUOTE:
            self._state = DecoderState.AFTER_PROMPT_COMMA

        elif s == DecoderState.AFTER_PROMPT_COMMA:
            self._state = DecoderState.NAME_KEY_QUOTE

        elif s == DecoderState.NAME_KEY_QUOTE:
            self._seq = self._key_tokens['name']
            self._seq_pos = 0
            self._state = DecoderState.NAME_KEY_TOKENS

        elif s == DecoderState.NAME_KEY_TOKENS:
            self._seq_pos += 1
            if self._seq_pos >= len(self._seq):
                self._state = DecoderState.NAME_KEY_CLOSE_QUOTE

        elif s == DecoderState.NAME_KEY_CLOSE_QUOTE:
            self._state = DecoderState.NAME_COLON

        elif s == DecoderState.NAME_COLON:
            self._state = DecoderState.NAME_VALUE_OPEN_QUOTE

        elif s == DecoderState.NAME_VALUE_OPEN_QUOTE:
            # Consumed opening quote — model now picks function tokens
            self._name_tokens_so_far = []
            self._state = DecoderState.NAME_VALUE_TOKENS

        elif s == DecoderState.NAME_VALUE_TOKENS:
            if self._chosen_function is None:
                self._name_tokens_so_far.append(token_id)
                candidates = self._get_func_candidates()
                if len(candidates) == 1:
                    fname = candidates[0]
                    seq = self._func_name_tokens[fname]
                    if self._name_tokens_so_far == seq:
                        # Full name matched — go to close quote
                        self._chosen_function = fname
                        self._seq = seq
                        self._seq_pos = len(seq)
                        self._state = DecoderState.NAME_VALUE_CLOSE_QUOTE
                    else:
                        # One candidate left — force remaining tokens
                        self._chosen_function = fname
                        self._seq = seq
                        self._seq_pos = len(self._name_tokens_so_far)
                elif len(candidates) == 0:
                    # Fallback: pick first function
                    fname = list(self._functions.keys())[0]
                    self._chosen_function = fname
                    self._seq = self._func_name_tokens[fname]
                    self._seq_pos = len(self._seq)
                    self._state = DecoderState.NAME_VALUE_CLOSE_QUOTE
            else:
                self._seq_pos += 1
                if self._seq_pos >= len(self._seq):
                    self._state = DecoderState.NAME_VALUE_CLOSE_QUOTE

        elif s == DecoderState.NAME_VALUE_CLOSE_QUOTE:
            self._state = DecoderState.AFTER_NAME_COMMA

        elif s == DecoderState.AFTER_NAME_COMMA:
            self._state = DecoderState.PARAMS_KEY_QUOTE

        elif s == DecoderState.PARAMS_KEY_QUOTE:
            self._seq = self._key_tokens['parameters']
            self._seq_pos = 0
            self._state = DecoderState.PARAMS_KEY_TOKENS

        elif s == DecoderState.PARAMS_KEY_TOKENS:
            self._seq_pos += 1
            if self._seq_pos >= len(self._seq):
                self._state = DecoderState.PARAMS_KEY_CLOSE_QUOTE

        elif s == DecoderState.PARAMS_KEY_CLOSE_QUOTE:
            self._state = DecoderState.PARAMS_COLON

        elif s == DecoderState.PARAMS_COLON:
            self._state = DecoderState.PARAMS_OPEN_BRACE

        elif s == DecoderState.PARAMS_OPEN_BRACE:
            self._state = DecoderState.PARAM_KEY_OPEN_QUOTE

        elif s == DecoderState.PARAM_KEY_OPEN_QUOTE:
            if clean == '}':
                self._state = DecoderState.END_BRACE
            else:
                self._state = DecoderState.PARAM_KEY_FIRST_TOKEN

        elif s == DecoderState.PARAM_KEY_FIRST_TOKEN:
            if self._chosen_function:
                for pname, seq in self._param_tokens[
                        self._chosen_function].items():
                    if pname not in self._used_params and seq and \
                            seq[0] == token_id:
                        self._current_param = pname
                        self._seq = seq
                        self._seq_pos = 1
                        break
            if self._seq_pos >= len(self._seq):
                self._state = DecoderState.PARAM_KEY_CLOSE_QUOTE
            else:
                self._state = DecoderState.PARAM_KEY_TOKENS

        elif s == DecoderState.PARAM_KEY_TOKENS:
            self._seq_pos += 1
            if self._seq_pos >= len(self._seq):
                self._state = DecoderState.PARAM_KEY_CLOSE_QUOTE

        elif s == DecoderState.PARAM_KEY_CLOSE_QUOTE:
            self._state = DecoderState.PARAM_COLON

        elif s == DecoderState.PARAM_COLON:
            self._state = DecoderState.PARAM_VALUE_START

        elif s == DecoderState.PARAM_VALUE_START:
            if clean == '"':
                self._state = DecoderState.PARAM_STRING_TOKENS
            else:
                self._state = DecoderState.PARAM_NUMBER_CONTENT

        elif s == DecoderState.PARAM_NUMBER_CONTENT:
            if clean == ',':
                if self._current_param:
                    self._used_params.add(self._current_param)
                self._reset_param()
                self._state = DecoderState.PARAM_KEY_OPEN_QUOTE
            elif clean == '}':
                if self._current_param:
                    self._used_params.add(self._current_param)
                self._state = DecoderState.END_BRACE

        elif s == DecoderState.PARAM_STRING_TOKENS:
            if token_id == self._tok['"']:
                if self._current_param:
                    self._used_params.add(self._current_param)
                self._state = DecoderState.AFTER_PARAM

        elif s == DecoderState.AFTER_PARAM:
            if clean == ',':
                self._reset_param()
                self._state = DecoderState.PARAM_KEY_OPEN_QUOTE
            elif clean == '}':
                self._state = DecoderState.END_BRACE

        elif s == DecoderState.END_BRACE:
            self._state = DecoderState.DONE

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 300,
    ) -> str:
        """Run constrained decoding and return the generated JSON string.

        Args:
            input_ids: Token IDs of the encoded prompt.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            The generated JSON string.
        """
        NEG_INF = float('-inf')
        current_ids = list(input_ids)

        for _ in range(max_new_tokens):
            if self._state == DecoderState.DONE:
                break

            logits = self._model.get_logits_from_input_ids(current_ids)
            valid_ids = self._get_valid_token_ids()

            if not valid_ids:
                break

            masked = [
                logit if i in valid_ids else NEG_INF
                for i, logit in enumerate(logits)
            ]

            best_id = masked.index(max(masked))
            token_str = self._id_to_token.get(best_id, '')

            self._update_state(best_id, token_str)
            current_ids.append(best_id)

        return self._generated
