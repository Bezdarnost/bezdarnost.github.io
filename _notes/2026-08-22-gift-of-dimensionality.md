---
title: High-dimensional input is hard; high-dimensional parameters are easy
date: 2026-08-22
kind: tweet
featured: true
tags:
  - overparameterization
  - generalization
  - optimization
aphorism:
  lines:
    - "High-dimensional <em>input</em> makes modeling hard."
    - "High-dimensional <em>parameter space</em> makes model estimation easy."
  by: Yann LeCun
facts:
  - value: input dim
    label: modeling is hard
  - value: param dim
    label: estimation is easy
  - value: overparam
    label: can still generalize
papers:
  - title: Reconciling modern machine-learning practice and the bias-variance trade-off
    authors: Belkin, Hsu, Ma, Mandal
    year: 2019
    url: https://arxiv.org/abs/1812.11118
source_name: Yann LeCun on X
source_url: https://x.com/ylecun/status/1701315853550014843
---

[Greg Brockman](https://x.com/gdb) called the curse of dimensionality a misnomer — billion-dimensional spaces are why neural nets train at all, maybe a "gift of dimensionality." The original line is not the part worth keeping.

[Yann LeCun](https://x.com/ylecun)'s correction is. High-dimensional _input_ makes modeling hard. High-dimensional _parameter space_ makes model estimation easy. People have known the second fact for a long time: ADMM and EM add auxiliary variables so optimization gets easier; kernel methods put one parameter per training sample and can fit whatever you want, generalization depending on the kernel.

That bigger nets are easier to train — and that local minima mostly go away — is an old intuition. Theories came later. The idea that had a hard time becoming mainstream is the one that contradicted every statistics textbook: a widely over-parameterized net can still _generalize_ well.
