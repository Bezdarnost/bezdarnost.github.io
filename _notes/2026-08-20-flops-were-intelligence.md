---
title: FLOPs were intelligence; parameters were knowledge
date: 2026-08-20
kind: tweet
featured: true
tags:
  - scaling laws
  - MoE
  - Switch Transformers
aphorism:
  lines:
    - 'FLOPs were <em>intelligence</em>;'
    - 'parameters were <em>knowledge</em>!'
  by: Noam Shazeer
  via: Liam Fedus
facts:
  - value: 1 of 2048
    label: experts per token
  - value: "<3B"
    label: activated params
  - value: 1.6T
    label: total params
papers:
  - title: Switch Transformers
    authors: Fedus, Zoph, Shazeer
    year: 2021
    url: https://arxiv.org/abs/2101.03961
source_name: Liam Fedus on X
source_url: https://x.com/LiamFedus/status/2090363702042304847
---

Prompted by [Jie Tang](https://x.com/jietang)'s history of scaling laws, [Liam Fedus](https://x.com/LiamFedus) went back to the Switch Transformer era.

In 2020, they explored the limits of sparsity by routing each token to only 1 out of 2048 experts — in retrospect, a bold choice. The model had fewer than 3B activated parameters, but 1.6T total parameters, comparable to today's frontier models.

The 1.6T model achieved better C4 perplexities than the T5 models using far less compute, set a new SOTA on TriviaQA, but was *dumb as bricks* on reasoning tasks like SuperGLUE.

The lesson: the optimal tokens-per-parameter ratio is highly task-dependent.
