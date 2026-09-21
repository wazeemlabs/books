"""A question set with the one label a cache measurement needs.

To measure whether a cache served the *right* answer, something has to
know which questions share an answer and which only look as though they
do. No public trace carries that label, so this set carries it: every
question belongs to a topic, every topic has one answer, and two
questions share an answer exactly when they share a topic.

Topics are grouped into families. Within a family the questions are
written to be confusable -- they differ by one decisive word: adult or
child, over or under, before or after, covered or not covered. Those
are the pairs a cache must keep apart, and they are the pairs every
similarity measure scores highest. Across families the questions are
simply unrelated.

Within a topic the paraphrases are deliberately of two kinds:

* near-identical -- a typo, a capital, a missing question mark. These
  are the hits an exact-match cache misses for no good reason and a
  normalizer recovers for free.
* rewritten -- the same need in different words, sharing almost no
  vocabulary. These are the hits only a semantic cache can get, and
  they are the ones that force the threshold down to where the
  confusable pairs start being matched too.

The set is authored, not sampled, and Chapter 32 says so where it uses
it. What it supports is a statement about the *shape* of the problem:
that the same-answer and different-answer similarity distributions
overlap, and that they overlap because of how questions are written
rather than because of any weakness in the measure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topic:
    """One question, however it is asked, and the answer it must get."""

    id: str
    family: str
    answer: str
    asked: tuple[str, ...]


TOPICS: tuple[Topic, ...] = (
    # --- dosing: the decisive word is who it is for -------------------
    Topic("dose-adult", "dose", "One 500 mg tablet every eight hours.", (
        "Is the 500 mg dose safe for an adult?",
        "is the 500mg dose safe for an adult",
        "Can a grown-up take 500 mg of this safely?",
        "What is the right amount for someone over eighteen?",
    )),
    Topic("dose-child", "dose", "Do not give this to anyone under twelve.", (
        "Is the 500 mg dose safe for a child?",
        "is the 500mg dose safe for a child",
        "Can I give 500 mg of this to my son?",
        "What is the right amount for a nine year old?",
    )),

    # --- refunds: the decisive word is the direction of a threshold ---
    Topic("refund-over", "refund", "Orders above 100 dollars are refunded in full.", (
        "What is the refund policy for orders over $100?",
        "what is the refund policy for orders over 100 dollars",
        "I spent more than a hundred, do I get my money back?",
        "How are large purchases handled if I return them?",
    )),
    Topic("refund-under", "refund", "Orders of 100 dollars or less get store credit.", (
        "What is the refund policy for orders under $100?",
        "what is the refund policy for orders under 100 dollars",
        "I spent less than a hundred, do I get my money back?",
        "How are small purchases handled if I return them?",
    )),

    # --- warranty: the decisive word is a negation --------------------
    Topic("warranty-yes", "warranty", "Yes, manufacturing faults are covered.", (
        "Is this covered by the warranty?",
        "is this covered by the warranty",
        "Will you repair a fault in the product for free?",
        "Does the guarantee pay for this kind of damage?",
    )),
    Topic("warranty-no", "warranty", "No, accidental damage is not covered.", (
        "Is this not covered by the warranty?",
        "Is this excluded from the warranty?",
        "Which kinds of damage do you refuse to repair?",
        "What falls outside the guarantee?",
    )),

    # --- cancelling: the decisive word is a point in time -------------
    Topic("cancel-before", "cancel", "Cancel before renewal and you are not charged.", (
        "What happens if I cancel before my renewal date?",
        "what happens if i cancel before my renewal date",
        "I want to stop being billed next month, what do I do?",
        "How do I end the plan while I still have time?",
    )),
    Topic("cancel-after", "cancel", "After renewal the term is charged in full.", (
        "What happens if I cancel after my renewal date?",
        "what happens if i cancel after my renewal date",
        "I was billed yesterday and want out, what now?",
        "Can I get out of a term that has already started?",
    )),

    # --- shipping: the decisive word is where it is going -------------
    Topic("ship-domestic", "shipping", "Two working days, free above 50 dollars.", (
        "How long does domestic shipping take?",
        "how long does domestic shipping take",
        "When will my parcel arrive if I live in the same country?",
        "What is delivery like for local orders?",
    )),
    Topic("ship-international", "shipping", "Ten to fifteen working days, duties payable.", (
        "How long does international shipping take?",
        "how long does international shipping take",
        "When will my parcel arrive if I live abroad?",
        "What is delivery like for orders going overseas?",
    )),

    # --- leave: the decisive word is what kind of worker --------------
    Topic("leave-staff", "leave", "Twenty paid days a year, accruing monthly.", (
        "How many days of leave does a full-time employee get?",
        "how many days of leave does a full time employee get",
        "What is the holiday allowance for permanent staff?",
        "As a salaried worker, what time off am I owed?",
    )),
    Topic("leave-contractor", "leave", "Contractors accrue no paid leave.", (
        "How many days of leave does a contractor get?",
        "how many days of leave does a contractor get",
        "What is the holiday allowance for agency staff?",
        "On a day rate, what time off am I owed?",
    )),

    # --- deposits: the decisive word is one prefix --------------------
    Topic("deposit-refundable", "deposit", "Returned within ten days of checkout.", (
        "Is the deposit refundable?",
        "is the deposit refundable",
        "Do I get the security money back when I leave?",
        "What happens to the money I put down at the start?",
    )),
    Topic("deposit-nonrefundable", "deposit", "The booking fee is kept whatever happens.", (
        "Is the deposit non-refundable?",
        "is the deposit nonrefundable",
        "Do you keep the booking money if I do not turn up?",
        "Which part of what I paid can I never get back?",
    )),

    # --- rates: the decisive word is one adjective --------------------
    Topic("rate-fixed", "rate", "Fixed at 6.1 per cent for the whole term.", (
        "What is the rate on a fixed loan?",
        "what is the rate on a fixed loan",
        "If I lock the price in, what do I pay?",
        "What does it cost to borrow with no change over time?",
    )),
    Topic("rate-variable", "rate", "Tracks base plus 1.4 per cent and moves.", (
        "What is the rate on a variable loan?",
        "what is the rate on a variable loan",
        "If I let the price move, what do I pay?",
        "What does it cost to borrow when the rate can change?",
    )),

    # --- unrelated topics, for the traffic to have a tail -------------
    Topic("password", "account", "Use the reset link on the sign-in page.", (
        "How do I reset my password?",
        "how do i reset my password",
        "I cannot get in to my account, what should I do?",
        "Forgot my login details, help",
    )),
    Topic("invoice", "billing", "Invoices are in Billing, under Documents.", (
        "Where can I download my invoice?",
        "where can i download my invoice",
        "I need a receipt for my accountant",
        "How do I get a copy of what I was charged?",
    )),
    Topic("hours", "contact", "Nine to five on working days, local time.", (
        "What are your opening hours?",
        "what are your opening hours",
        "When is someone actually there to answer?",
        "Are you open on Saturday?",
    )),
    Topic("address", "contact", "Change it under Settings, then Addresses.", (
        "How do I change my delivery address?",
        "how do i change my delivery address",
        "I have moved house and need my parcels sent elsewhere",
        "Where do I update where things get sent?",
    )),
    Topic("tracking", "shipping-status", "The tracking link is in your dispatch email.", (
        "How do I track my order?",
        "how do i track my order",
        "Where is my parcel right now?",
        "Can I see where my delivery has got to?",
    )),
    Topic("vat", "billing", "Prices shown include VAT at the standard rate.", (
        "Do your prices include VAT?",
        "do your prices include vat",
        "Is tax added at the checkout?",
        "Is the number I see the number I pay?",
    )),
)


def by_id() -> dict[str, Topic]:
    return {t.id: t for t in TOPICS}


def questions() -> list[tuple[str, Topic]]:
    """Every way every topic is asked, with the topic it belongs to."""
    return [(q, t) for t in TOPICS for q in t.asked]


def test_every_topic_has_a_partner_or_is_alone_on_purpose() -> None:
    """A family with one member measures nothing: it has nothing to be
    confused with. Either a topic has a confusable sibling, or its
    family is the tail of unrelated questions."""
    from collections import Counter
    sizes = Counter(t.family for t in TOPICS)
    confusable = [f for f, n in sizes.items() if n > 1]
    assert len(confusable) >= 8, sizes
    assert sum(sizes[f] for f in confusable) >= 16, sizes


def test_answers_are_unique_per_topic() -> None:
    """Two topics sharing an answer string would make a wrong hit look
    right, and the measurement would quietly flatter every cache."""
    answers = [t.answer for t in TOPICS]
    assert len(set(answers)) == len(answers)
    assert len({t.id for t in TOPICS}) == len(TOPICS)


def test_each_topic_is_asked_more_than_one_way() -> None:
    for t in TOPICS:
        assert len(t.asked) >= 3, t.id
        assert len(set(t.asked)) == len(t.asked), t.id
