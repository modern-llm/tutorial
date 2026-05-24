---
episode: 10
slug: dpo
title: "DPO and preference optimization"
core_question: "Closed-form simplification of RLHF; the descendants (KTO, IPO, ORPO)"
target_minutes: 15
voice: en-US-AndrewMultilingualNeural
---

# Episode 10: D P O and preference optimization

Welcome to episode ten. Last time we walked through the full R L H F pipeline. S F T, then a reward model, then P P O. Three stages, four models in memory, online sampling, finicky hyperparameters. It works. Chat G P T proved that. But it is expensive, complex, and operationally painful. In twenty twenty three, a paper from Stanford asked whether you could get the same result with a single supervised loss and no reward model at all. The paper is called Direct Preference Optimization, or D P O, by Rafailov, Sharma, Mitchell, and colleagues. It is one of those rare papers that is both theoretically elegant and practically useful, and it reshaped how the field thinks about preference alignment.

Let me build up to the D P O loss step by step, because the derivation is the interesting part.

The starting point is the standard R L H F objective from episode nine. You have a policy pi, which is the language model you are training. You have a reference policy pi ref, which is the frozen S F T model. You have a reward function r. The R L H F objective is to find the policy that maximizes the expected reward minus beta times the K L divergence from the reference policy. In words, you want the policy that gets the highest reward while staying close to the reference.

This is a constrained optimization problem. And it turns out it has a closed-form solution. You can write down the optimal policy exactly. For any response y given a prompt x, the log probability under the optimal policy is the log probability under the reference policy plus one over beta times the reward, minus a normalizing constant Z of x that depends on the prompt but not on the specific response.

In equation form. Log pi star of y given x equals log pi ref of y given x plus one over beta times r of x, y, minus log Z of x.

This is the optimal policy. If you could compute it directly, you would not need P P O at all. The problem is the normalizing constant Z. It is a sum, or integral, over all possible responses, weighted by the reference policy and the reward. For language models with vocabularies of tens of thousands of tokens and responses of hundreds of tokens, this is intractable to compute.

But here is the key move. Rearrange the equation to solve for the reward instead of the policy. From the equation above, the reward is beta times the difference in log probabilities plus beta times log Z. That is, r of x, y equals beta times log pi star of y given x minus beta times log pi ref of y given x plus beta times log Z of x.

Now recall from episode nine how the reward model was trained. It uses the Bradley-Terry model. Given a prompt x, the probability that response y w is preferred over response y l is the sigmoid of the reward difference, r of x, y w minus r of x, y l. Substitute the expression for r that we just derived. When you take the difference of the rewards, the log Z terms cancel, because Z depends only on x, not on the response. What remains is sigma of beta times the log ratio for the chosen response minus beta times the log ratio for the rejected response, where the log ratio is log pi over pi ref.

That is the D P O loss. It depends only on the policy model and the reference model, evaluated on preference pairs. No reward model. No Z. No intractable normalization.

Let me state it cleanly. The D P O loss for a single preference pair is the negative log sigmoid of beta times a quantity. That quantity is the log probability of the chosen response under the current policy minus the log probability of the chosen response under the reference, minus the same thing for the rejected response. In shorthand, you are measuring how much the policy has shifted probability toward the chosen response and away from the rejected response, relative to where the reference started.

Two forward passes per preference pair. One through the current policy. One through the frozen reference. Compute the log probabilities of both the chosen and rejected responses under both models. Plug into the loss. Backpropagate. Update the policy. That is the entire training loop.

No reward model to train or maintain. No online sampling. No P P O with its clip ranges and value heads. No advantage estimation. No generation during training. Just a supervised loss on existing preference data.

Let me make the intuition concrete. The loss says, relative to the reference model, make the chosen response more likely and the rejected response less likely. The "relative to the reference" part is critical. If the loss just said "make the chosen response likely," the model would memorize the chosen responses. By measuring the shift from the reference, the loss captures the direction of preference rather than the absolute quality. A response that the reference already ranks highly does not need much further boosting. A response that the reference ranks low but that was chosen by the human needs a big shift.

Beta is the temperature parameter. It controls how aggressive the optimization is. Low beta, say zero point one, means each preference pair produces a small update. The policy stays close to the reference. High beta, say zero point five or one, means larger updates, and the policy can deviate more. In practice, beta values between zero point one and zero point five work for most tasks. The sensitivity is low. People who spent months tuning P P O hyperparameters found that D P O needed almost no tuning.

The practical workflow becomes dramatically simpler. Start with your S F T model. Freeze a copy as the reference. Get a dataset of preference pairs. Each example is a prompt, a chosen response, and a rejected response. Fine-tune the S F T model with the D P O loss on those pairs. Done. You go from three stages and four models to one stage and two models, where one is frozen and only does forward passes.

The training is stable. The loss is a smooth, well-behaved function. There are no policy gradients, which are inherently high-variance. No ratio clipping. No moving reward baselines. No value function that needs to track a non-stationary target. People who had spent months debugging P P O runs found that D P O converged in a few hours of training with minimal hyperparameter search.

Now, is D P O strictly better than R L H F? No. There are real tradeoffs, and understanding them is important.

The biggest limitation is that D P O is offline. The preference data is fixed before training starts. The model trains on a frozen set of human comparisons and never generates its own responses to explore. In P P O, the model samples fresh responses at each step, and the reward model scores them. If the model discovers a novel response pattern that scores well, it gets reinforced. If it drifts into a bad region, the reward model catches it immediately. This online exploration means P P O can discover good behaviors that are not represented in any fixed dataset.

D P O cannot do this. If the preference dataset does not contain examples of sophisticated multi-step reasoning, D P O cannot learn it. It can only learn to prefer what was already preferred in the data it was given. For tasks where the space of possible good responses is large and the training data covers only a fraction of it, D P O is limited.

This matters most on hard reasoning and math tasks. If the preference dataset contains responses at a certain quality level, D P O can learn to rank them, but it cannot learn to produce responses that are better than anything in the dataset. Online R L methods, like P P O or the G R P O we will cover in the next episode, can discover superior behaviors through exploration, especially when paired with verifiable rewards.

A second limitation is sensitivity to label noise. Human preferences are noisy. Sometimes the chosen response is not actually better. Maybe the labeler was tired, or the difference was marginal, or the labeler had a different interpretation of quality. P P O is somewhat robust to this because the reward model generalizes. It smooths over individual label errors by learning a continuous scoring function from many examples. D P O trains directly on the pairs, so a mislabeled pair gives exactly one wrong gradient signal.

There is also a more subtle theoretical issue. The D P O derivation assumes the policy is at the optimum of the R L H F objective. The closed-form solution for the reward, which makes Z cancel, only holds exactly at the optimal policy. During training, the policy is not at the optimum, it is on its way there. So the equivalence between D P O and R L H F is approximate during training. Some researchers have argued this gap matters in practice, leading to underfitting compared to proper online R L.

Despite these limitations, D P O was adopted widely and quickly. Llama two, released in mid twenty twenty three, used R L H F with P P O. Llama three, released in twenty twenty four, switched to D P O for its preference-tuning stage. That switch at Meta tells the story. The simplicity advantage is so large that even if D P O gives up a few points on the hardest benchmarks, the reduction in engineering complexity, training time, compute cost, and debugging effort makes it the better choice for most teams.

D P O also spawned a family of descendants, each addressing one of its limitations. Let me walk through the important ones.

I P O, Identity Preference Optimization. Published by Azar and colleagues at DeepMind shortly after D P O. It addresses the overfitting problem. In D P O, as training progresses, the log probabilities of rejected responses can be driven to negative infinity. The loss keeps pushing them down without bound, especially for easy preference pairs where the distinction is clear. This is overfitting in the preference space. I P O modifies the loss to directly regularize the gap between chosen and rejected log probability ratios, preventing it from growing without limit. The effect is a smoother, more stable training dynamic, especially with noisy or imbalanced preference data.

K T O, Kahneman-Tversky Optimization. Published by Ethayarajh and colleagues in twenty twenty four. K T O removes the need for paired preferences entirely. D P O requires pairs, a chosen and a rejected response to the same prompt. K T O works with unpaired binary feedback. "This response was good." "That response, to a completely different prompt, was bad." No pairing needed. The loss function is inspired by prospect theory from behavioral economics, hence the Kahneman-Tversky name, and it weighs losses more heavily than gains, reflecting the empirical observation that humans are more sensitive to bad outputs than to marginally better good ones.

This is practically significant because unpaired feedback is much easier to collect at scale. Users give thumbs up or thumbs down on individual responses. They almost never see two responses side by side. Building a system that collects paired preferences requires showing users two responses and asking them to compare, which is a worse user experience and yields far less data. K T O lets you use the natural feedback signal that users already generate.

O R P O, Odds Ratio Preference Optimization. Published in twenty twenty four. This combines the S F T stage and the preference stage into a single training phase. Instead of first doing S F T to get the format right and then doing D P O to align preferences, O R P O trains on both objectives simultaneously. The intuition is that instruction following and preference alignment push in the same direction. You want the model to produce good responses to instructions, and you want it to prefer better responses over worse ones. O R P O saves a training stage and avoids the risk of catastrophic forgetting that can happen when you do S F T followed by a separate preference stage.

There are others worth knowing by name. S L i C H F uses rejection sampling to construct preference pairs on the fly. S P I N uses the model's own generations as negative examples against the S F T data as positives, creating a self-play dynamic. Sim P O simplifies D P O further by removing the reference model entirely, using a length-normalized implicit reward instead. The common thread is the insight that D P O established. You do not need an explicit reward model. Preference learning can be baked into the language model's own probability space through a well-designed loss function.

Let me frame this in the arc of the series. Episode nine described the full R L H F pipeline, the three-stage, four-model system that made Chat G P T possible. This episode described D P O, which collapsed the reward model stage entirely and replaced P P O with a supervised loss. The simplification was real, large, and practically important. D P O is still the most common preference-tuning method in production.

But D P O's limitation, offline preferences only, leaves the door open for something else. What if you have a problem where you can verify the answer automatically? Math, where the answer is either correct or not. Code, where it either passes the test suite or does not. In that case, you do not need human preferences at all. You can score responses programmatically, sample many of them, and do R L directly against verifiable outcomes. That is G R P O, Group Relative Policy Optimization, and it is the subject of the next episode.

Thanks for listening. See you next time.
