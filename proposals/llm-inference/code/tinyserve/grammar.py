"""Deciding, before the model speaks, which tokens it is allowed to say.

A model asked for JSON usually produces JSON. Usually is not a
contract, and a service whose caller parses the reply needs one.

Constrained decoding supplies it by changing what happens between the
logits and the sampler. At every step, work out which tokens could
legally come next given what has been emitted so far, set the logits
of all the others to negative infinity, and sample from what is left.
The model cannot emit a token that breaks the format because the token
is not available to it.

That requires two things: a description of the format precise enough
to answer "what may come next", and a way to turn that answer into a
mask over the vocabulary quickly enough to do it for every token. This
module builds both for JSON, which needs a stack -- brackets nest --
and so is the smallest interesting case.

Written for `LLM Inference from the Ground Up`, Part VI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

# A vocabulary of JSON pieces. A real tokenizer's pieces are learned
# from text and cut across the structure -- one token can be `":"` or
# `"},{"` -- which is what makes the compilation in production harder
# than this. The shape of the problem is the same.
VOCAB: list[str] = [
    "{", "}", "[", "]", ",", ":",
    '"', "true", "false", "null",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", ".",
    "a", "b", "c", "d", "e", "i", "l", "m", "n", "o", "r", "s", "t", "u",
    " ", "<eos>",
]
TOKEN_ID = {t: i for i, t in enumerate(VOCAB)}
EOS = TOKEN_ID["<eos>"]
DIGITS = [t for t in VOCAB if t.isdigit()]
LETTERS = [t for t in VOCAB if t.isalpha() and len(t) == 1]
LITERALS = ["true", "false", "null"]


@dataclass(frozen=True)
class JsonMachine:
    """Where we are in a JSON document, and what may come next.

    The state is a label, a stack of the containers still open, and
    three facts about the thing being written: whether a string is a
    key or a value, and whether a number has seen a digit or a dot
    yet. That stack is why a regular expression cannot do this job:
    JSON nests to any depth, and a machine with finitely many states
    cannot remember how many brackets it owes.
    """

    label: str = "start"
    stack: tuple[str, ...] = ()
    in_key: bool = False          # the string being written is a key
    seen_digit: bool = False      # the number so far ends in a digit,
                                  # so it could stop here -- "1." cannot
    seen_dot: bool = False        # ... and has had its decimal point
    leading_zero: bool = False    # the integer part is exactly "0", so
                                  # no further digit may follow it

    text: str = ""

    def key(self) -> tuple:
        """The machine's full state: everything but the text so far."""
        return (self.label, self.stack, self.in_key, self.seen_digit,
                self.seen_dot, self.leading_zero)

    def mask_key(self) -> tuple:
        """Only what the *mask* depends on.

        Much less than the full state. What may come next depends on
        the innermost open container, never on the ones beneath it: a
        comma inside an object means the same thing three levels down
        as it does at the top. So the whole stack collapses to its last
        element, and the number of distinct masks stops growing with
        depth even though the number of states does not.

        This is the observation an engine's grammar compiler is built
        on, and Chapter 31 measures what it is worth.
        """
        return (self.label, self.top, self.in_key, self.seen_digit,
                self.seen_dot, self.leading_zero)

    @property
    def top(self) -> str | None:
        return self.stack[-1] if self.stack else None

    def _close(self) -> list[str]:
        """The bracket that closes the innermost container, if any."""
        return ["}"] if self.top == "object" else ["]"] if self.top else []

    def allowed(self) -> list[str]:
        """The tokens that would keep this document legal."""
        s = self.label
        if s == "start":
            return ["{", "["]
        if s == "string":
            # Any character, or the quote that ends it.
            return LETTERS + DIGITS + [" ", '"']
        if s == "obj_open":
            return ['"', "}"]           # a key, or an empty object
        if s == "colon":
            return [":"]
        if s == "value":
            return ["{", "[", '"', "-"] + DIGITS + LITERALS
        if s == "arr_open":
            return ["{", "[", '"', "-", "]"] + DIGITS + LITERALS
        if s == "number":
            # JSON has no leading zeros: "0" and "0.5" are numbers,
            # "094" is not, so once the integer part is a lone zero the
            # only ways on are a decimal point or the end of it.
            out = [] if (self.leading_zero and not self.seen_dot) else list(DIGITS)
            if self.seen_digit and not self.seen_dot:
                out.append(".")
            if self.seen_digit:      # false right after "-" or "."
                out += [","] + self._close()
            return out
        if s == "after_value":
            return [","] + self._close()
        if s == "comma":
            if self.top == "object":
                return ['"']            # only a key may follow in an object
            return ["{", "[", '"', "-"] + DIGITS + LITERALS
        if s == "done":
            return ["<eos>"]
        raise ValueError(f"unknown state {s!r}")

    def step(self, token: str) -> "JsonMachine":
        """Advance, assuming `token` is one `allowed` offered.

        Branching on the state first and the token second, which is
        the only safe order. Branching on the token first reads more
        naturally and is wrong: a digit means one thing inside a
        string and another inside a number, and a machine that decides
        what a digit means before asking where it is will quietly
        accept `{"4}` as a document.
        """
        stack, s, text = list(self.stack), self.label, self.text + token

        # Inside a string, everything is a character until the quote.
        if s == "string":
            if token != '"':
                return JsonMachine(label="string", stack=self.stack,
                                   in_key=self.in_key, text=text)
            after = ("colon" if self.in_key
                     else ("done" if not stack else "after_value"))
            return JsonMachine(label=after, stack=self.stack, text=text)

        # Inside a number, digits and one dot continue it; anything
        # else ends it and has to be handled as if the number were
        # already complete.
        if s == "number" and (token in DIGITS or token == "."):
            # "-0" leads with a zero just as "0" does.
            lead = (self.leading_zero
                    or (token == "0" and not self.seen_digit
                        and not self.seen_dot))
            return JsonMachine(
                label="number", stack=self.stack, in_key=self.in_key,
                # A dot makes the number unfinished again: JSON has no
                # "1." and the mask has to stop offering a close.
                seen_digit=token in DIGITS,
                seen_dot=self.seen_dot or token == ".",
                leading_zero=lead, text=text)

        if token == "{":
            stack.append("object")
            return JsonMachine("obj_open", tuple(stack), text=text)
        if token == "[":
            stack.append("array")
            return JsonMachine("arr_open", tuple(stack), text=text)
        if token in ("}", "]"):
            stack.pop()
            label = "done" if not stack else "after_value"
            return JsonMachine(label, tuple(stack), text=text)
        if token == '"':
            # Opening a string. It is a key exactly when an object is
            # waiting for one.
            is_key = s in ("obj_open", "comma") and self.top == "object"
            return JsonMachine("string", self.stack, in_key=is_key, text=text)
        if token == ":":
            return JsonMachine("value", self.stack, text=text)
        if token == ",":
            return JsonMachine("comma", self.stack, text=text)
        if token in LITERALS:
            label = "done" if not stack else "after_value"
            return JsonMachine(label, self.stack, text=text)
        if token in DIGITS or token == "-":
            return JsonMachine("number", self.stack,
                               seen_digit=token in DIGITS,
                               leading_zero=token == "0", text=text)
        raise ValueError(f"token {token!r} is not legal in state {s!r}")


def mask_for(machine: JsonMachine) -> np.ndarray:
    """A boolean over the vocabulary: True where the token is allowed."""
    out = np.zeros(len(VOCAB), dtype=bool)
    for token in machine.allowed():
        out[TOKEN_ID[token]] = True
    return out


def apply(logits: np.ndarray, machine: JsonMachine) -> np.ndarray:
    """Logits with every disallowed token removed.

    Negative infinity rather than zero, because this happens *before*
    the softmax: a token at negative infinity gets probability zero
    exactly, and the rest are renormalised over what remains.
    """
    out = np.array(logits, dtype=np.float64)
    out[~mask_for(machine)] = -np.inf
    return out


def reachable_states(max_depth: int = 3) -> list["JsonMachine"]:
    """Every state a document of bounded depth can reach.

    This is the number that makes constrained decoding cheap. The mask
    depends only on the state, so if the states can be enumerated in
    advance, the per-token cost at serving time is a lookup rather
    than a walk over the grammar.

    Returned as machines with no text, since the text is deliberately
    not part of a state's identity.
    """
    seen: dict[tuple, JsonMachine] = {}
    frontier = [JsonMachine()]
    while frontier:
        m = frontier.pop()
        bare = JsonMachine(m.label, m.stack, m.in_key, m.seen_digit,
                           m.seen_dot, m.leading_zero)
        if bare.key() in seen or len(m.stack) > max_depth:
            continue
        seen[bare.key()] = bare
        for token in m.allowed():
            if token == "<eos>":
                continue
            frontier.append(bare.step(token))
    return sorted(seen.values(), key=lambda m: (len(m.stack), m.label))


def mask_table(max_depth: int = 3) -> dict[tuple, np.ndarray]:
    """Every mask, built once.

    What an engine compiles a grammar into. At serving time the work
    per token is looking one of these up and adding it to the logits;
    the grammar is not consulted again.
    """
    return {m.mask_key(): mask_for(m) for m in reachable_states(max_depth)}


def generate(logits_for, machine: JsonMachine | None = None,
             max_tokens: int = 200, rng=None, constrained: bool = True
             ) -> dict:
    """Sample a document, with the mask applied or not.

    Returns the text, whether it finished, whether it parses, and how
    many tokens it took. Finishing and parsing are different questions
    and the chapter needs both: a walk cut off at `max_tokens` is
    incomplete, which is not the same failure as one that ended and
    was wrong.

    `logits_for` is handed the text so far and the machine's current
    state, and returns logits over the vocabulary; it stands in for the
    model. A real model sees only the text, but a stand-in that is
    meant to be *good at* the format needs to know where in the format
    it is, and passing the state is the honest way to give it that
    rather than letting it read a stale one.
    """
    rng = rng or np.random.default_rng(0)
    m = machine or JsonMachine()
    for n in range(1, max_tokens + 1):
        logits = np.asarray(logits_for(m.text, m), dtype=np.float64)
        if constrained:
            logits = apply(logits, m)
        shifted = logits - logits.max()
        probs = np.exp(shifted)
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            return {"text": m.text, "finished": False, "parses": False,
                    "tokens": n, "stopped": "no token allowed"}
        probs = probs / total
        token = VOCAB[int(rng.choice(len(VOCAB), p=probs))]
        if token == "<eos>":
            return {"text": m.text, "finished": m.label in ("done", "start"),
                    "parses": parses(m.text), "tokens": n,
                    "stopped": "end of text"}
        if not constrained:
            # No machine at all: this is what decoding does when
            # nobody is checking. The text accumulates and whether it
            # is JSON is discovered at the end, by the caller, too late.
            m = JsonMachine(label=m.label, stack=m.stack,
                            text=m.text + token)
            continue
        m = m.step(token)
        if m.label == "done":
            return {"text": m.text, "finished": True,
                    "parses": parses(m.text), "tokens": n,
                    "stopped": "document complete"}
    return {"text": m.text, "finished": False, "parses": parses(m.text),
            "tokens": max_tokens, "stopped": "ran out of tokens"}


def parses(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


# --- tests: the specification -------------------------------------------


def test_a_constrained_walk_is_always_valid_json() -> None:
    """The contract. Every document the machine allows must parse.

    Walked with a uniform random choice among the allowed tokens,
    which is the most hostile sampler there is: it has no idea what
    JSON looks like and relies entirely on the mask.
    """
    rng = np.random.default_rng(0)
    flat = np.zeros(len(VOCAB))          # a model with no opinion at all
    bad, finished = [], 0
    for _ in range(600):
        r = generate(lambda _t, _m: flat, rng=rng, constrained=True,
                     max_tokens=400)
        if not r["finished"]:
            continue                 # cut off, not wrong: a separate case
        finished += 1
        if not r["parses"]:
            bad.append(r["text"])
    assert finished > 100, f"only {finished} walks finished; test is vacuous"
    assert not bad, (f"{len(bad)} of {finished} finished constrained walks "
                     f"did not parse; first was {bad[0]!r}")


def test_without_the_mask_it_is_not_valid() -> None:
    """The comparison that makes the mask worth its cost.

    The same sampler with no mask should almost never produce JSON. If
    it did, the chapter would have no subject.
    """
    rng = np.random.default_rng(1)
    flat = np.zeros(len(VOCAB))
    good = sum(generate(lambda _t, _m: flat, rng=rng, constrained=False,
                        max_tokens=120)["parses"] for _ in range(400))
    assert good < 40, f"{good} of 400 unconstrained walks parsed"


def test_the_mask_never_allows_nothing() -> None:
    """A state with no legal continuation would deadlock the sampler.

    Every reachable state must offer at least one token, or a request
    that reaches it hangs until its timeout -- the worst failure a
    serving path can have, because it consumes a slot and produces
    nothing.
    """
    for m in reachable_states(max_depth=3):
        assert m.allowed(), f"state {m.key()} allows no token at all"
        assert mask_for(m).any(), m.key()


def test_the_states_are_few_enough_to_precompute() -> None:
    """The claim that makes this cheap at serving time.

    If the number of states grew with the document, a mask would have
    to be computed per token. It grows only with nesting depth, so the
    masks can be built once and looked up.
    """
    depths = (1, 2, 3, 4, 5, 6)
    states = [len(reachable_states(d)) for d in depths]
    masks = [len(mask_table(d)) for d in depths]

    # The states really do explode: the stack is a sequence of objects
    # and arrays, so there are twice as many of them at each new level.
    assert states[-1] > 2 * states[-3], states

    # The masks do not, because a mask depends on the innermost
    # container and not on the ones beneath it. If this ever starts
    # growing, something has crept into `mask_key` that does not
    # belong there, and the chapter's whole argument goes with it.
    assert len(set(masks)) == 1, (
        f"the number of distinct masks should not depend on depth, "
        f"got {masks} for depths {depths}")
    assert masks[0] < 40, f"{masks[0]} masks is more than expected"
