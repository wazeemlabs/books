# Glossary

Every technical term the book uses, in plain words, with the chapter
that introduces it. This is Appendix I, and it is also a contract:
`make check` fails if any term below is used in a chapter earlier than
the one that defines it.

Format: `term | defined in chapter | definition`. A chapter claims a
term by carrying a `<!-- defines: term -->` marker beside its
explanation.

| Term | Ch | What it means |
|---|---|---|
| inference | 1 | Using a trained model to answer a question. The opposite of training, which is making the model in the first place. This whole book is about inference. |
| token | 1 | The unit of text a model reads and writes: a common chunk, roughly three-quarters of a word in English. Models never see letters. |
| parameter | 1 | One number, learned during training and fixed thereafter. A model's behaviour is entirely these numbers. "8 billion parameters" means 8 billion of them. |
| weight | 1 | Another word for parameter. Used when emphasising that the number must be fetched from memory to be used. |
| bandwidth | 1 | How many bytes per second a machine can move from memory into the processor. Distinct from latency, which is how long one fetch takes to arrive. |
| prefill | 1 | Reading the prompt. All its tokens are known, so they can be processed at once. |
| decode | 1 | Writing the reply, one token at a time, each waiting for the one before it. |
| KV cache | 1 | The stored keys and values of every token so far, kept so they need not be recomputed. The single largest consumer of memory in serving. |
| latency | 1 | How long one thing takes. In serving, always qualified: time to first token, or time between tokens. |
| throughput | 1 | How many tokens per second a whole server produces, across every user. |
| batching, batch | 1 | Serving several sequences in the same pass over the weights, so the cost of fetching them is shared. The batch is the set of sequences advanced together in one pass. |
| precision | 1 | How many bytes each number is stored in. Fewer bytes means less to fetch and less exactness. |
| quantization | 1 | Deliberately storing a model's numbers in fewer bytes, to fetch fewer of them. |
| allocator | 1 | The part of a system that hands out memory and takes it back. |
| utilization | 4 | The fraction of the time a machine is doing useful work. The term that usually decides whether self-hosting is cheaper than an API. |
| provenance | 1 | The record of what produced a number: the machine, the software versions, the date, the commit. A measurement without it cannot be reproduced. |
| vector | 2 | A list of numbers. That is all. |
| matrix | 2 | A rectangle of numbers. Multiplying matrices is where a model spends nearly all its time. |
| tensor | 2 | A list, rectangle or higher-dimensional block of numbers. A vector and a matrix are both tensors. |
| embedding | 2 | The vector a token is replaced by, looked up from a table. |
| attention | 2 | The step in which each token looks at the tokens before it and collects information from them. |
| query, key, value | 2 | The three vectors each token produces. A query is what it is looking for, a key is what it offers to anyone looking, a value is what it hands over when looked at. |
| causal mask | 2 | The rule, built into the architecture, that a token may only look at tokens at or before it — never forwards. |
| head | 2 | One independent copy of attention. A layer runs several at once, each with its own queries, keys and values. |
| feed-forward | 2 | The half of a layer in which each token is processed on its own, with no communication between tokens. Most of a model's parameters live here. |
| residual | 2 | The running representation each token carries through the layers, which each layer adds to rather than replaces. |
| normalization | 2 | Rescaling a vector so its numbers stay in a workable range. Necessary for deep models to train at all. |
| logits | 2 | The raw scores a model produces, one for every token in its vocabulary, before they are turned into probabilities. |
| softmax | 2 | The step that turns raw scores into probabilities that add up to one. |
| vocabulary | 2 | The fixed list of tokens a model knows. Real ones hold 30,000 to 200,000. |
| tokenizer | 9 | The program that cuts text into tokens and back again. |
| autoregressive | 11 | Generating one token at a time, each conditioned on everything written so far. Why decoding cannot be parallelised across a single answer. |
| transformer | 2 | The architecture every model in this book uses: a stack of layers, each doing attention and then feed-forward. |
| greedy decoding | 2 | Always taking the highest-scoring next token. Deterministic, which is why the book uses it when two runs must match. |
| sampling | 2 | Choosing the next token at random in proportion to its score, rather than always taking the best. Why a chatbot varies. |
| arithmetic intensity | 3 | How much arithmetic an operation performs per byte it must fetch. The number that decides which limit it hits. |
| memory-bound | 3 | Limited by how fast bytes arrive, not by how fast arithmetic can be done. Decoding is memory-bound. |
| compute-bound | 3 | Limited by arithmetic rather than by fetching. Prefill is compute-bound. |
| time to first token | 3 | How long a user waits before anything appears. Governed by prefill. |
| inter-token latency | 3 | The gap between successive words once they start. Governed by decode. |
| grouped-query attention | 3 | Letting several query heads share one key-value head, so the KV cache is a fraction of the size. Every production model does this. |
| multi-query attention | 3 | The extreme of grouped-query attention: one key-value head for all queries. |
| last-level cache | 4 | Small fast memory near the processor, which helps only if data is reused. Decoding reuses nothing, which is why it does not help. |
| memory wall | 1 | The widening gap between how fast processors compute and how fast memory delivers. Named in 1995; serving a language model is what hitting it looks like. |
| HBM | 4 | High-bandwidth memory: the fast memory attached to an accelerator. Where a model's weights live. |
| FLOP | 4 | One floating-point operation. FLOP/s is how many per second. |
| mixture-of-experts | 4 | A model that activates only a fraction of its parameters for each token, attacking the memory wall directly. |
| percentile | 5 | A value below which a given share of measurements fall. The p99 is what the unluckiest request in a hundred experiences. |
| goodput | 5 | Throughput that actually met the promise. Throughput that arrived too late to be useful counts for nothing. |
| offered load | 5 | The work arriving at a server, whether or not it can keep up. Throughput is what came out; offered load is what was asked for. When the second exceeds the first, the queue grows and every latency number becomes a function of how long the test ran. |
| service level objective | 5 | The stated promise a service makes about its latency and availability. Turns engineering into a constrained problem. |
| concurrency | 1 | How many sequences are being served at the same time. |
| queueing | 5 | Requests waiting because the server is busy. The reason a service degrades sharply rather than gradually. |
| streaming multiprocessor | 7 | One of the many independent processing units inside an accelerator. An H100 has 132. |
| kernel | 3 | A program that runs on an accelerator. Also, loosely, one such operation. |
| interconnect | 7 | The link between accelerators. Fast inside a machine, much slower between machines. |
| occupancy | 7 | How full each processing unit is kept. Distinct from utilization, and much harder to be satisfied by. |
| driver | 7 | The software layer between the operating system and the accelerator. Its version affects results, so it is recorded. |
| spot instance | 7 | A rented machine that is cheaper because it can be taken away at short notice. Fine for measurement, not for serving. |
| roofline | 6 | A chart bounding what a machine can achieve, given an operation's arithmetic per byte. |
| break-even point | 3 | The arithmetic-per-byte at which a machine stops being limited by memory and starts being limited by arithmetic. |
| BLAS | 9 | The standard library of matrix routines every numerical program relies on. |
| warmup | 9 | Running work before timing it, so the measurement is not of first-run effects. |
| median | 5 | The middle value. What the book reports, because an average is dragged by the tail. |
| closed loop, open loop | 9 | Two ways to generate load. A closed loop slows its own clients when the server slows, so it can never reveal overload; an open loop sends at a fixed rate regardless. |
| poisson process | 9 | The standard model of independent arrivals: a given average rate, with the gaps between arrivals varying rather than evenly spaced. It is what makes "twelve requests a second" bursty, and the bursts are what overload a server. |
| differential testing | 10 | Checking one implementation against an independent one, rather than against a test written by the same author. |
| baseline | 9 | The recorded "before" measurement, with provenance, that later work is compared against. |
| contiguous | 13 | Occupying one unbroken run of memory. The requirement that paging removes. |
| fragmentation | 12 | Memory held but unusable. Internal fragmentation is space reserved inside a slot and never filled. |
| over-reservation | 13 | Committing memory for a length nobody knows yet, because the allocator must commit before the answer is written. |
| block, block table | 14 | A fixed-size unit of KV storage, and the per-sequence list saying which blocks it owns and in what order. |
| paging | 3 | Storing a sequence's cache in fixed-size blocks that can be anywhere, so nothing need be reserved in advance. |
| indirection | 14 | Reaching data through a table of addresses rather than directly. The cost paging pays for its flexibility. |
| eviction, preemption | 12 | Taking memory back from a running sequence when the pool fills, and recomputing or restoring it later. |
| scheduler | 12 | The part of a server that decides, over and over while it runs, which requests are worked on next and which wait. Not the model, and not the allocator; the thing that calls both. |
| prefix | 3 | However much of a prompt's start another prompt also had. The unit prefix caching reuses. |
| prefix caching | 3 | Reusing the keys and values of a prompt's opening because some earlier request already computed them. |
| working set | 4 | The amount of data a piece of work actually touches, as opposed to how much it could touch. |
| radix tree, prefix tree | 6 | An index that groups items by how they begin, so everything sharing an opening is found in one walk down from the root. |
| system prompt | 11 | The standing instruction an application puts in front of whatever the user typed. Identical on every request, which is what makes it worth caching. |
| reference count | 14 | A tally of how many holders a piece of memory has. It is released only when the last of them lets go. |
| cache hit, cache miss | 15 | Whether what was asked for was already there. |
| hit rate | 15 | The share of what was asked for that was already there. For a prefix cache, measured in prompt tokens, not in requests. |
| hash | 15 | A short number computed from a piece of data, used as its name. Two different pieces of data are overwhelmingly unlikely to get the same one. |
| least recently used | 15 | The eviction rule that throws away whatever has gone longest without being wanted. |
| least frequently used | 15 | The eviction rule that throws away whatever has been wanted least often. Blind to age, which is its weakness. |
| copy-on-write | 15 | Sharing something until somebody writes to it, and copying only then. Prefix caching avoids needing it by sharing only blocks that are already full. |
| side channel | 15 | Information leaked by how long something takes rather than by what it returns. |
| continuous batching, iteration-level scheduling | 17 | Deciding which sequences are in the batch before every forward pass instead of once. Arrivals join at the next iteration; finished sequences leave immediately and free their slot. |
| admission control | 17 | The decision about when to let new work into a system, as distinct from what to do with the work already in it. For a serving engine: when to start a waiting request's prefill. |
| head-of-line blocking | 17 | Work stuck behind unrelated work in front of it, purely because of the order the two were put in. A static batch is head-of-line blocking by design. |
| watermark | 17 | Memory a scheduler keeps free rather than handing out, so that the blocks freed by an eviction cannot immediately be spent re-admitting the sequence that was evicted. |
| chunked prefill | 3 | Reading a prompt over several iterations instead of one, so that it never stops the sequences already decoding for longer than one chunk takes. |
| stall-free batching | 18 | A schedule in which decoding never pauses: every running sequence takes its token first, and prefill gets whatever is left of the iteration's token budget. |
| token budget | 18 | The cap on how many token positions one iteration may carry, across decodes and prefill chunks together. The knob that decides the wait between tokens. |
| shortest-job-first | 18 | Serving the shortest waiting work first. It minimises the average wait and lengthens the longest one, which is the whole of the fairness argument. |
| swapping | 17 | Making room by copying a sequence's cache to host memory and back, rather than discarding it and computing it again. |
| disaggregation | 19 | Running prefill and decode on different machines, and moving the keys and values between them. |
| tensor parallelism | 19 | Splitting each weight matrix across several accelerators so they share the work of one layer. A way to fit a model that does not fit, and to go faster once it does. |
| design decision record | 19 | A short document naming a decision, what else was considered, the measurement that settled it, and the condition that would reverse it. One closes each Part of this book. |
| score matrix | 20 | One number for every pair of a query position and a key position. It grows with the square of the sequence length, which is what makes attention expensive to move. |
| SRAM | 20 | The small, fast scratchpad attached to each core of an accelerator, a few hundred kilobytes, against the tens of gigabytes of HBM everything shares. |
| kernel launch | 20 | One instruction from the host processor telling the accelerator to run one program. Results pass between launches through HBM, which is why fusing two kernels into one saves traffic. |
| online softmax | 20 | Finishing a softmax without having seen all of its inputs, by carrying a running maximum and rescaling what is already accumulated whenever the maximum moves. Exact, not approximate. |
| mantissa | 22 | The digits of a floating-point number, as distinct from its scale. More mantissa bits means more digits survive. |
| exponent | 22 | The scale of a floating-point number: the power of two the mantissa is multiplied by. More exponent bits means a wider range of sizes. |
| accumulator | 22 | The running total a dot product adds into. Its format need not match the inputs', and on a tensor core it does not: small inputs, a float32 total. |
| symmetric quantization | 24 | Integer codes spread evenly either side of zero, so zero maps to code zero exactly. What weights use. |
| asymmetric quantization | 24 | Integer codes spanning the actual range from smallest to largest, with a zero point recording which code means zero. What one-sided data uses. |
| scale | 24 | The size of one step of an integer code: the largest magnitude in a group divided by the largest code. Shared by every value in the group. |
| zero point | 24 | The code that stands for zero in an asymmetric scheme. Stored alongside the scale, one per group. |
| draft model | 29 | The cheap model that guesses the next few tokens for a more expensive one to check. It must be cheap and it must agree; an unrelated model agrees at chance. |
| target model | 29 | The expensive model whose answers you actually want. Speculative decoding produces exactly its distribution, whatever the draft does. |
| acceptance rate | 29 | How often a draft's guess survives verification. The one input that decides whether speculation pays, and the only one that has to be measured rather than derived. |
