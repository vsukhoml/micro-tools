import math
import numpy as np
from scipy.stats import binom
from scipy.special import logsumexp

def critbinom_logspace(num_trials: int, neg_log_prob: float, neg_log_alpha_compl: float) -> int:
  """
  Calculates the critical value of the binomial distribution with high precision
  using log-space arithmetic and a binary search.
  It can be used for calculating values for the NIST SP 800-90B APT test.

  This is a replacement for `scipy.stats.binom.isf` when the quantile
  is extremely close to 0 and doesn't fit into `double` type used by Python
  (i.e., neg_log_alpha_compl is larger than 53).

  Args:
    num_trials: The number of Bernoulli trials (n).
    neg_log_prob: The negative log base 2 of the probability of success (p).
    neg_log_alpha_compl: The negative log base 2 of the complement of the
                         criterion value alpha.

  Returns:
    The smallest integer k that satisfies the condition.
  """
  # Edge case: If the criterion is >= 1, the answer is always num_trials
  if neg_log_alpha_compl <= 0:
    return num_trials

  p = math.pow(2, -neg_log_prob)

  # Handle trivial probabilities
  if p == 0:
      return 0
  if p == 1:
      return num_trials

  # This is our target value on the log scale: log(2**(-C3))
  log_q = -neg_log_alpha_compl * math.log(2)

  # Binary search for the smallest k in [0, num_trials]
  low = 0
  high = num_trials
  ans = num_trials # Default answer if no k satisfies the condition until the end

  dist = binom(n=num_trials, p=p)

  while low <= high:
    k = (low + high) // 2

    if k >= num_trials:
        # P(X > k) is 0, so log(P(X > k)) is -inf. This always satisfies the condition.
        log_survival_prob = -np.inf
    else:
        # Calculate log(P(X > k)) = log(sum_{i=k+1 to n} P(X=i))
        i_values = np.arange(k + 1, num_trials + 1)
        log_probabilities = dist.logpmf(i_values)
        log_survival_prob = logsumexp(log_probabilities)

    if log_survival_prob <= log_q:
      # This k works. Let's see if an even smaller k also works.
      ans = k
      high = k - 1
    else:
      # This k is too small, P(X > k) is too large. We need a larger k.
      low = k + 1

  return ans

# --- Example Usage ---
trials = 1024
prob_exp = 0.5
alpha_exp = 54 # A large value that causes the standard `isf` to fail

for alpha_exp in range(50, 55):
	# 1. Using the previous SciPy isf-based approach (which fails)
	p = math.pow(2, -prob_exp)
	q = math.pow(2, -alpha_exp) # This is a very small number
	k_scipy_fails = binom.isf(q, trials, p)
	print(f"SciPy's isf approach with alpha={alpha_exp}: k = {int(k_scipy_fails)}")

	# 2. Using the robust log-space binary search
	k_robust = critbinom_logspace(
	    num_trials=trials,
	    neg_log_prob=prob_exp,
	    neg_log_alpha_compl=alpha_exp
	)
	print(f"Robust log-space approach with alpha={alpha_exp}: k = {k_robust}")
