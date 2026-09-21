| Tokens made worse | By how much | Mean loss moves | Perplexity moves | Replies with at least one |
|---|---|---|---|---|
| 0.1% | 0.5 nats | 0.0005 | +0.05% | **26%** |
| 0.1% | 1.0 nats | 0.0010 | +0.10% | **26%** |
| 0.1% | 2.0 nats | 0.0020 | +0.20% | **26%** |
| 0.1% | 4.0 nats | 0.0040 | +0.40% | **26%** |
| 1.0% | 0.5 nats | 0.0050 | +0.50% | **95%** |
| 1.0% | 1.0 nats | 0.0100 | +1.01% | **95%** |
| 1.0% | 2.0 nats | 0.0200 | +2.02% | **95%** |
| 1.0% | 4.0 nats | 0.0400 | +4.08% | **95%** |
| 5.0% | 0.5 nats | 0.0250 | +2.53% | **100%** |
| 5.0% | 1.0 nats | 0.0500 | +5.13% | **100%** |
| 5.0% | 2.0 nats | 0.1000 | +10.52% | **100%** |
| 5.0% | 4.0 nats | 0.2000 | +22.14% | **100%** |

Exact arithmetic. Perplexity is the exponential of a mean log-loss over tokens, so a change confined to a small share of tokens moves it by the product and no more. The last column is the same change seen the way a user sees it: the chance that a reply of 300 tokens contains at least one of them. One token in a thousand is invisible in the first measure and in a quarter of replies in the second.
