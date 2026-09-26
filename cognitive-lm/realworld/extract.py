"""Open fact extraction from corpus passages: the checksum's contents.

An instruct model reads one passage and answers the question from that
passage alone, or NONE. It never sees the candidates, so the facts it
extracts are the same whatever the answering model proposes: this builds
the (subject, relation, value) set an offline pipeline would, restricted to
the subjects we ask about.

A strong extractor may answer from its own memory instead of the passage.
extract_cached is also run on passages about a different subject (the
control in experiments/real_extract.py) to measure how often that happens.
"""

import hashlib
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
NONE = "NONE"
PROMPT = ("Passage:\n{passage}\n\n"
          "Using only the passage above, answer: {question}\n"
          "Answer only if the passage states it about this exact subject. "
          "If it does not, answer NONE. Reply with the answer alone, a few words at most.")


def key(qid, passage):
    return f"{qid}|{hashlib.blake2b(passage.encode(), digest_size=8).hexdigest()}"


class Extractor:
    def __init__(self, model=MODEL, device=None, dtype=torch.bfloat16):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        # eager attention: SDPA on MPS returns NaN for left-padded prompts (see generate.py)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, attn_implementation="eager").to(self.device).eval()

    @torch.no_grad()
    def run(self, pairs, max_new_tokens=16):
        """pairs: (question, passage). Returns the extracted answer or NONE for each."""
        texts = [self.tok.apply_chat_template([{"role": "user", "content": PROMPT.format(passage=p, question=q)}],
                                              tokenize=False, add_generation_prompt=True) for q, p in pairs]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                  pad_token_id=self.tok.pad_token_id)
        ans = [self.tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
               for o in out]
        return [NONE if a.strip(" .").upper() == NONE or not a.strip(" .") else a.strip(" .") for a in ans]


def extract_cached(make_extractor, jobs, path, batch=8, log=print):
    """jobs: (qid, question, passage). Returns {key(qid, passage): answer}.
    make_extractor is called (loading the model) only if something is not cached."""
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                done[r["key"]] = r["answer"]
    todo = [j for j in jobs if key(j[0], j[2]) not in done]
    ex = make_extractor() if todo else None
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        res = ex.run([(q, p) for _, q, p in chunk])
        with open(path, "a") as f:
            for (qid, _, p), a in zip(chunk, res):
                done[key(qid, p)] = a
                f.write(json.dumps({"key": key(qid, p), "answer": a}) + "\n")
        if (s // batch) % 25 == 0:
            log(f"extracted {min(s + batch, len(todo))}/{len(todo)}")
    return done
