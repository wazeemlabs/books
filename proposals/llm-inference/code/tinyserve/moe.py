"""A model whose weights are mostly not used, and what a batch does to that.

A mixture-of-experts layer holds many feed-forward networks and sends
each token to a few of them. The selling point is in the headline
numbers: DeepSeek-V3 is "671B total parameters with 37B activated for
each token", so a decode step reads the weights of a 37B model and
answers with the quality of a much larger one.

That is true of one token. A server does not decode one token.

Every sequence in the batch routes independently, so the step has to
read the *union* of the experts any of them chose, and the union grows
fast: with 256 experts and 8 per token, a batch of 128 touches almost
all of them. The weights a step reads therefore climb from the
activated count toward the total count as the batch grows, which is
the opposite of how a dense model behaves and is the single fact that
decides how a mixture of experts is served.

Reference: DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv:2412.19437.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoE:
    """A mixture-of-experts model, from its published numbers.

    `total` and `active` are the two figures every such model is
    announced with. `routed` and `per_token` are how many experts each
    layer holds and how many of them a token goes to. Everything else
    here is derived from those four.
    """

    name: str
    total: int
    active: int
    routed: int
    per_token: int
    layers: int
    d_model: int

    def expert_params(self) -> float:
        """Parameters in one routed expert, solved from the headline pair.

        The activated count is everything a single token reads: the
        parts that are always read, plus `per_token` experts. The total
        is the same always-read parts plus all `routed` of them. Two
        equations, two unknowns.
        """
        if self.routed <= self.per_token:
            raise ValueError("a mixture needs more experts than it routes to")
        return (self.total - self.active) / (self.routed - self.per_token)

    def always_read(self) -> float:
        """Parameters every token reads whatever it routes to: attention,
        embeddings, shared experts, any dense layers."""
        return self.active - self.per_token * self.expert_params()

    def touched(self, batch: int) -> float:
        """Expected distinct experts a batch of `batch` tokens reaches.

        Each token picks `per_token` of `routed`, so an expert is
        missed by one token with probability `1 - per_token / routed`
        and by all of them with that raised to the batch. Uniform
        routing is assumed, which is what the load-balancing machinery
        in these models is for; skewed routing touches fewer experts at
        a small batch and the same number at a large one.
        """
        if batch < 0:
            raise ValueError("a batch cannot be negative")
        miss = 1.0 - self.per_token / self.routed
        return self.routed * (1.0 - miss ** batch)

    def params_read(self, batch: int) -> float:
        """Parameters one decode step has to fetch for this batch."""
        return self.always_read() + self.expert_params() * self.touched(batch)

    def params_per_token(self, batch: int) -> float:
        """The same, shared across the tokens the step produces."""
        return self.params_read(batch) / max(batch, 1)

    def dense_equivalent(self, batch: int) -> float:
        """The dense model whose decode step would read the same bytes.

        The number to quote when someone says a mixture of experts
        "runs like a 37B model": at what batch, and what does it run
        like at yours?
        """
        return self.params_read(batch)


def step_seconds(m: MoE, batch: int, seq: int, hbm_bytes_per_s: float,
                 kv_bytes_per_token: int, elem_bytes: int = 2) -> float:
    """One decode step, under the book's roofline.

    Same shape as a dense step: the weights the step actually touches,
    once, plus each sequence's own cached keys and values. The only
    difference is that the first term now depends on the batch.
    """
    weights = m.params_read(batch) * elem_bytes
    cache = batch * seq * kv_bytes_per_token
    return (weights + cache) / hbm_bytes_per_s


def params_read_per_machine(m: MoE, batch: int, machines: int) -> float:
    """What one machine fetches when the experts are spread across many.

    Expert parallelism gives each machine a share of the routed
    experts. It still holds the parts every token needs -- attention,
    embeddings, shared experts -- because every token needs them, so
    that part is replicated and read in full on every machine. Only the
    routed experts divide.

    This is why a mixture of experts is not a single-machine model.
    The weights a step reads grow with the batch, and the only thing
    that keeps the step fast is that they are divided by the machines
    at the same time.
    """
    if machines < 1:
        raise ValueError("a fleet has at least one machine")
    return m.always_read() + m.expert_params() * m.touched(batch) / machines


def all_to_all_bytes(m: MoE, batch: int, elem_bytes: int = 2) -> float:
    """Bytes crossing the network per decode step under expert parallelism.

    With the experts spread across machines, a token whose chosen
    experts live elsewhere has to send its hidden state there and
    receive the result back. Each token sends `per_token` copies of a
    `d_model` vector out and gets `per_token` back, once per MoE layer.
    This is the traffic the interconnect has to carry while the step
    runs, and Chapter 19 measured what the links can do.
    """
    per_layer = 2 * batch * m.per_token * m.d_model * elem_bytes
    return per_layer * m.layers


# --- the models this chapter uses ---------------------------------------

# DeepSeek-V3: "671B total parameters with 37B activated for each
# token"; 61 layers, hidden 7168, one shared and 256 routed experts
# with 8 activated per token (FACTS.md).
DEEPSEEK_V3 = MoE(name="DeepSeek-V3", total=671_000_000_000,
                  active=37_000_000_000, routed=256, per_token=8,
                  layers=61, d_model=7168)


# --- tests ---------------------------------------------------------------


def test_the_headline_pair_is_recovered() -> None:
    """One token must read the activated count, and every expert the total."""
    m = DEEPSEEK_V3
    assert abs(m.params_read(1) - m.active) / m.active < 1e-9
    # Every expert touched is the whole model, near enough that the
    # difference is the experts a large batch still misses.
    whole = m.always_read() + m.expert_params() * m.routed
    assert abs(whole - m.total) / m.total < 1e-9


def test_a_batch_touches_more_experts_than_a_token() -> None:
    m = DEEPSEEK_V3
    assert m.touched(1) == m.per_token
    assert m.touched(8) > m.touched(1)
    assert m.touched(1024) > m.routed * 0.999
    assert m.touched(0) == 0.0


def test_the_step_grows_and_the_token_gets_cheaper() -> None:
    """The two things a batch does to a mixture, which point opposite ways.

    A bigger batch makes the step read more weights -- so the wait
    between tokens gets worse -- while sharing them over more tokens,
    so the cost per token gets better. Both at once, and confusing them
    is how a mixture gets deployed badly.
    """
    m = DEEPSEEK_V3
    assert m.params_read(128) > 4 * m.params_read(1)
    assert m.params_per_token(128) < m.params_per_token(1) / 4


def test_spreading_the_experts_divides_only_the_experts() -> None:
    """The replicated part does not shrink, which is what sets the floor
    on how far this can be taken."""
    m = DEEPSEEK_V3
    one = params_read_per_machine(m, 128, 1)
    many = params_read_per_machine(m, 128, 16)
    assert many < one
    assert many > m.always_read()
    huge = params_read_per_machine(m, 128, 10_000)
    assert abs(huge - m.always_read()) / m.always_read() < 0.01


def test_expert_parallel_traffic_scales_with_the_batch() -> None:
    m = DEEPSEEK_V3
    assert all_to_all_bytes(m, 2) == 2 * all_to_all_bytes(m, 1)
    assert all_to_all_bytes(m, 0) == 0
