"""The observation side of E02: what a reported number says about the world it came from.

`bins` holds the one bounded-bin Student-t kernel of the stage. A water cut in the history
and a saturation in a log are both reported ROUNDED and both live on the unit interval, so
neither is a point observation and neither is unbounded: the likelihood of a report is the
probability of the bin it was rounded into, under a Student-t on the bounded g-space
`g(f) = arcsin(sqrt(f))`, renormalised to `[0, pi/2]`.

The watercut history (plan §4), the saturation logs (§5) and the noise recovery (§11) all
score through that one kernel. A second implementation of the same probability would be a
second likelihood, and two likelihoods that disagree in the tail are two different
posteriors wearing one name.
"""
