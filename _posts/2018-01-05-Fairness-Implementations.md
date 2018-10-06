---
values:
  comments: true
title: Fairness Implementations
tags:
  project
---
TL;DR I built a [repository of fairness algorithm implementations](https://github.com/yo-shavit/fairml-farm).

There are no good "fairness" benchmarks that we all agree on.
That is, there is no universally agreed-upon mathematically quantifiable target of "fairness" that we as a community aim to optimize (yet).

Many migrants from the machine learning community, myself included, find this baffling.
ML researchers have basically reached a consensus on their shared benchmarks (top-k accuracy, or [BLEU](https://en.wikipedia.org/wiki/BLEU) or reward in benchmark environments).
Members of the ML community can directly compare different algorithms' performances to each other, and unleash the potential of many individuals working in parallel on a well-specified problem.

Not so for fairness. Instead, as 2018 dawns, the algorithmic fairness community is still trying to nail down a reasonable set of definitions for what is "fair".
That's absolutely a good thing: rigorizing complex social desires is a hard task, and I'm glad we're all making sure we get it right.
However, this also leads to many papers that follow a very specific, and troubling, format:
1. Propose a new notion of fairness
2. Construct an algorithm that optimizes decisions w.r.t. this new notion of fairness
3. Compare this algorithm to previous algorithms, which were not built with this notion of fairness in mind, and demonstrate your algorithm's superiority

Now, to some extent, this is fine. This is what we'd expect of a field focused primarily on finding definitions.
But in doing so, we are to some extent losing focus of the algorithms themselves.
That's unfortunate, because in my experience, comparing different fairness algorithms' behavior on the same set of data is one of the best ways to gain an intuitive understanding of the fairness definitions those algorithms are aiming to enforce.

To that end, I've begun compiling a set of fairness algorithm implementations [all in one place](https://github.com/yo-shavit/fairml-farm) so they can be easily compared and dissected.
The goal is to create a unified interface for training and evaluating algorithms on a wide swath of fairness metrics, so we can understand their tradeoffs not just [theoretically](https://arxiv.org/abs/1609.05807) but empirically.

For now, I've just included the "binary classification with single protected attribute" case, and have mostly kept to group notions of fairness.
You can play around with neural networks with different "fairness regularizers" (as seen in [this paper](https://arxiv.org/pdf/1707.00044.pdf)), and see how different hyperparameters improve or worsen accuracy and types of fairness.

I'm looking for feedback on the approach and design, so please shoot me your thoughts!
I've also intentionally made it really easy to add new algorithms/fairness datasets, so please submit pull-requests and add your own!

-Yo
