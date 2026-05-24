---
episode: 11
slug: grpo
title: "GRPO and verifiable rewards"
core_question: "Group-relative advantages, dropping the value head, why math/code unlocked RL"
target_minutes: 19
voice: en-US-AndrewMultilingualNeural
---

# Episode 11: G R P O and verifiable rewards

Welcome to episode eleven. Over the last two episodes we have walked through the evolution of post-training. Episode nine covered the full R L H F pipeline. S F T plus reward model plus P P O. Episode ten covered D P O, which collapsed the reward model and P P O into a single supervised loss. Both approaches ultimately depend on human preferences. Somebody, at some point, has to look at two responses and say which one is better.

But there is an entire class of tasks where you do not need a human to judge quality. Math. Code. Formal logic. Puzzle solving. Tasks where the answer is verifiable. Two plus two is four, or it is not. The code passes the test suite, or it does not. The proof is valid, or it is not. No reward model required. No human labeler required. You can just check.

This observation is what unlocked the next chapter. If you have verifiable rewards, you can do reinforcement learning at scale, without the overhead of training and maintaining a reward model, and without the limitations of an offline preference dataset. The algorithm that emerged to exploit this is G R P O, Group Relative Policy Optimization. It is the algorithm behind DeepSeek R one, the model that proved reasoning could be trained with R L, and it kicked off the wave of reasoning models in twenty twenty four and twenty twenty five.

Let me build up to G R P O by starting with what it replaced and why.

P P O, which we covered in episode nine, has several components. A policy model. A value head that estimates the expected future reward at each state. A reward model that scores responses. And a reference model for the K L penalty. The value head is the component I want to focus on, because removing it is G R P O's key innovation.

In P P O, the value head provides the baseline for computing advantages. The advantage for a response is the actual reward minus the value head's prediction. Why do you need a baseline at all? Because vanilla policy gradient, without a baseline, has very high variance. The gradient is proportional to the reward times the log probability of the action. If the reward is always positive, which it typically is for language tasks, every response gets its probability increased, just by different amounts. The signal about which responses are actually better than average is buried in noise. Subtracting a baseline that estimates the expected reward turns this into a signal about how much better or worse each response was compared to expectation. Positive advantage means better than expected, increase its probability. Negative means worse, decrease it.

In games and robotics, where P P O was designed, the value function works well because the state space has structure that makes prediction feasible. A chess position has a well-defined value. A robotic arm's joint angles correspond to a learnable value landscape. In language, the state is the entire conversation history up to that point, and predicting the expected reward from that state is nearly as hard as the language modeling task itself. The value head ends up being a large, expensive model that provides mediocre baseline estimates. It adds parameters, adds training complexity, and the gradient signal it contributes is noisy.

G R P O, proposed by the DeepSeek team in the DeepSeek Math paper in twenty twenty four, asks, what if we replace the value head with something much simpler?

Here is the G R P O algorithm step by step. For each prompt in the training batch, sample G completions from the current policy. G is the group size, typically eight or sixteen. Score each completion with the reward function. For verifiable tasks, this is just a correctness checker. Did the math answer match the ground truth? Did the code pass the tests? Each completion gets a reward, often binary, one for correct and zero for incorrect, though richer reward signals are possible.

Now compute the advantage for each completion within the group. Subtract the group mean reward and divide by the group standard deviation. The advantage for completion i is r sub i minus the mean of all r values in the group, divided by the standard deviation of the r values in the group. This is the "group relative" part of the name. The advantage is normalized within the group, not against a learned value function.

Then update the policy using a P P O style objective. For each token in each completion, compute the probability ratio between the current policy and the old policy that generated the completion. Clip the ratio to the range one minus epsilon to one plus epsilon, same as standard P P O. Multiply by the advantage. Add a K L penalty against a reference model. Maximize the objective.

That is the full algorithm. The value head is replaced by a group mean baseline. The reward model is replaced by a verifiable reward function. The ratio clipping and K L penalty are kept from P P O.

Let me explain why the group mean baseline works, because this is the core insight that makes G R P O viable. The purpose of any baseline in policy gradient is to reduce variance without introducing bias. Ideally, the baseline is the expected reward for the prompt. If the expected reward is zero point four, then a response with reward one has advantage zero point six, meaning better than expected, and a response with reward zero has advantage negative zero point four, meaning worse than expected. This is exactly the right signal.

The group mean, the average reward across G samples from the current policy for the same prompt, is an unbiased estimate of the expected reward. It is noisy, because G is finite, but the estimate improves with larger G. At G equals sixteen, you have sixteen samples, which gives a reasonable estimate for binary rewards. The standard deviation normalization further helps by making the advantage scale independent of the reward magnitude. If all sixteen samples happen to be correct, the advantages are all zero, which is the right thing, there is nothing to learn from a prompt where everything works.

There is a deeper point about why the value head is less important for language than for games. In classical R L, the value function serves two purposes. Variance reduction through baselining. And temporal credit assignment, figuring out which specific actions in a long trajectory were responsible for the eventual outcome. In language generation with a single end-of-sequence reward, temporal credit assignment is less critical. The reward reflects the entire generated response, and the policy gradient propagates through every token anyway. The main benefit of the value head is variance reduction, and the group mean provides that for free, without any additional parameters or training.

Now let me talk about verifiable rewards, because this is the second half of the story and equally important.

The reason G R P O works so well for math is not just the algorithm. It is that math provides a near-perfect reward signal. You give the model a problem. It generates a solution. You extract the final numerical answer, typically by parsing a structured output format, and compare it to the ground truth. Correct or not. One or zero. No noise in the reward. No ambiguity. No learned reward model that might be wrong or gameable.

Compare this to the general R L H F setting where the reward comes from a neural network trained on noisy human preferences. That reward model can be exploited. The policy can learn to produce responses that score high under the reward model but are actually bad. Verbose padding that inflates the reward model's length bias. Confident-sounding statements that exploit the reward model's preference for assertiveness. Technically wrong answers presented in a way that the reward model cannot distinguish from correct ones. With verifiable rewards, there is nothing to hack. The answer is right or it is wrong. The reward signal is incorruptible.

Code is the other major verifiable domain. Generate code, run it against a test suite, check the results. Some tasks offer richer reward signals than binary pass/fail. Partial credit for passing some tests but not others. Execution time or memory usage as secondary metrics. Compiler warnings as penalties. These richer signals provide more gradient information per sample.

There are also hybrid approaches where you combine verifiable rewards with format rewards. For example, you give a bonus for putting the answer inside a specific tag, or you penalize the model for producing responses that are too long. These shape the model's output format without requiring a learned reward model.

The combination of G R P O and verifiable rewards is what produced DeepSeek R one, released in January twenty twenty five. The training recipe, described in the R one paper, has several stages. Start with the DeepSeek V three base model. Do a cold start S F T on a small number of high-quality reasoning traces, to bias the model toward chain-of-thought formatting. Then run G R P O with verifiable math and code rewards. For each prompt, sample sixteen completions, score them, compute group-relative advantages, update.

What emerged during R L training was remarkable. Without being explicitly taught specific reasoning strategies, the model developed extended chain-of-thought behavior. It would write out its thinking step by step. It would catch its own errors. It would explicitly say things like "wait, that does not work, let me try a different approach" and then backtrack. It would verify its answer by re-deriving it a different way. This self-correction behavior was not in the S F T data, at least not in a prescriptive way. It was not in the reward function, which only checked the final answer. It emerged because the R L process discovered that generating longer, more careful reasoning chains led to more correct final answers. The reward gradient, flowing back through every token, strengthened whatever token patterns preceded correct answers. And those patterns turned out to look like explicit, structured reasoning.

The R one paper describes specific emergent phenomena during different phases of R L training. In the early phase, the model learns basic chain-of-thought decomposition. In the middle phase, it develops self-verification behaviors, re-reading its own work and checking for errors. In the later phase, it develops exploration behaviors, trying multiple approaches when the first one does not work. The researchers called this "the aha moment" when they first observed the model spontaneously generating "Wait! Let me reconsider..." without having been trained on examples of self-correction.

Let me add some important nuance to this clean narrative.

First, the cold start S F T matters more than it might seem. The R one paper emphasizes that without a few thousand examples of chain-of-thought reasoning in the S F T stage, the R L phase takes much longer to converge and can converge to less interpretable reasoning styles. The model might discover that certain degenerate token patterns, not meaningful to a human reader, happen to correlate with correct answers. The S F T stage biases the policy toward human-readable reasoning, making it easier for R L to refine and extend rather than invent from scratch.

Second, the K L penalty matters. Without it, the policy can drift far from coherent language. The model might find that repeating certain tokens or producing specific formatting artifacts correlates with correctness. The K L penalty keeps the policy close enough to the reference that its outputs remain natural, fluent language.

Third, the group size G affects the quality of training. Small G, say four, means a noisy estimate of the expected reward, which means high variance in the advantages, which means noisy gradients, which means slower or less stable learning. Large G, say sixty-four, means many samples per prompt, which is computationally expensive, because you are generating sixty-four full responses for every prompt in every batch. G equals sixteen is a common compromise that balances statistical quality against compute cost. The optimal G also depends on the reward distribution. For binary rewards, you need enough samples that the group includes both correct and incorrect responses. If the model is already ninety percent accurate on a problem, you need at least ten samples to expect even one incorrect response, and the advantage signals become sparse.

Fourth, the R one paper also applies R L on tasks beyond math. It uses code generation with automated test suites, and open-ended tasks with a separate reward model for non-verifiable domains. The verifiable-reward setting is the cleanest and produces the most dramatic results, but G R P O is not limited to verifiable tasks.

Let me now describe Karpathy's implementation in nanochat, because it illustrates how far you can strip this algorithm down while still getting results. Nanochat's chat R L script calls itself G R P O but removes several components compared to the full algorithm.

There is no K L penalty to a reference model. The trust region is deleted entirely. This is a stronger choice than it might appear. On-policy sampling reduces the need for P P O ratio machinery, since the policy that generated the data is the same as the one being updated. But K L also serves a separate purpose. It regularizes against language drift and reward hacking, preventing the model from finding degenerate token patterns that score well but are not coherent language. Nanochat gets away without it partly because the training runs are short and the model does not have time to drift far, and partly because the reward signal, binary math correctness, is hard to hack. On longer runs or with noisier rewards, dropping K L would be riskier.

There is no ratio clipping. Again, because the training is on-policy, the probability ratio between the new and old policy is always close to one if you do only one gradient step per batch of generated data. The ratio clipping in P P O is a safeguard against large policy changes, which is mainly needed when you do multiple gradient steps on the same batch of generated data.

The advantage does not divide by the standard deviation. It is just r minus the group mean, not r minus the group mean divided by the group standard deviation. The standard deviation normalization helps when rewards have varying scales across prompts, but for binary rewards on a consistent task like G S M eight K, the scale is naturally bounded and normalization is less important.

The loss is aggregated at the token level rather than the sequence level. Standard G R P O computes a single advantage per response and applies it uniformly to every token in that response. Nanochat follows a technique from the D A P O paper, by ByteDance, published in twenty twenty five, which computes the loss per token and averages across all tokens in the batch, rather than first averaging within each response and then across responses. The argument is that token-level aggregation gives more stable gradients on long generations, because it avoids giving disproportionate weight to short responses.

What is left is essentially REINFORCE with a group mean baseline. The simplest possible policy gradient algorithm applied to language generation. And it works. Karpathy demonstrates pass at k on G S M eight K improving steadily over training steps.

The lesson is striking. For on-policy training with verifiable rewards, most of the machinery in P P O, the value head, the ratio clipping, the K L penalty, the standard deviation normalization, can be removed without losing the core learning signal. The minimum viable R L algorithm for language is much simpler than the literature implies.

There is a lineage worth keeping in mind. P P O with all its components, designed for games and robotics. G R P O, which drops the value head and replaces it with group-relative advantages. Nanochat's variant, which further drops the K L penalty, the ratio clipping, and the standard deviation normalization. Each step removes a piece of machinery that was designed for a different setting where the state space, the action space, and the reward dynamics are very different from language.

Let me place this in the broader arc. Episode nine described R L H F, requiring human preferences, a reward model, and P P O. Episode ten described D P O, which simplified the pipeline by removing the reward model and P P O, at the cost of being offline. This episode describes G R P O, which goes back to online R L but avoids the reward model by using verifiable rewards, and avoids the value head by using group-relative advantages. Each step in the evolution removed something. D P O removed the reward model and P P O. G R P O removed the reward model and the value head. The field is converging on the simplest recipe that works for each setting.

In the next episode, we are going to look at what happens downstream of this training. When you train a model to reason with R L on verifiable rewards, something qualitatively new emerges. A model that gets better the longer it thinks. That is the test-time compute axis, and it opens up a new frontier of scaling that is independent of model size and training data.

Thanks for listening. See you next time.
