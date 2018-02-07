# ee-298-deep-learning
Learning Rate (LR) starts at 0.001.

LR is updated depending on the current normalized gradient (gradnorm) and the normalized previous gradient (prevgradnorm).

When prevgradnorm <= gradnorm, LR is decreased to 10% of the original LR (or decreased by 90%).

When prevgradnorm > gradnorm, LR is increased by 20%.

The tolerance (delta) used is 1.
