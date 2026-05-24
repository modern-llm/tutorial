---
episode: 16
slug: muon
title: "Optimizer renaissance: Adam W to Muon"
core_question: "Why the training stack is still alive; modded-nanogpt, Muon, schedules, and the next bottlenecks"
target_minutes: 18
voice: en-US-AndrewMultilingualNeural
---

# Episode 16: Optimizer renaissance, Adam W to Muon

Welcome to episode sixteen, the last episode of this series. We have covered a lot of ground. Attention. Scaling laws. Position embeddings. Inference efficiency. Mixture of experts. Long context. Lo R A. Multimodal models. The full arc from R L H F to D P O to G R P O. Reasoning models and test-time compute. Agents. Synthetic data. Evaluation. All of those are the loud revolutions, the ones that make headlines and launch products.

Today I want to talk about a quiet one. Optimizers. Specifically, why optimizers mattered again after years of being a solved problem, and what the Muon optimizer tells us about the free lunches still hiding in the training stack.

Let me start with the status quo that Muon disrupted. Since roughly twenty eighteen, the default optimizer for training transformers has been Adam W. To understand why Muon is interesting, you need to understand what Adam W does and what it misses.

Adam, proposed by Kingma and Ba in twenty fourteen, tracks two exponential moving averages for each parameter individually. The first moment estimate, m, tracks the running mean of the gradient. The second moment estimate, v, tracks the running mean of the squared gradient. The Adam update divides m by the square root of v, element-wise. This means each parameter gets its own adaptive learning rate. Parameters whose gradients are consistently large get smaller effective step sizes. Parameters whose gradients are small get larger effective step sizes. The result is that training is much less sensitive to the overall learning rate choice, because each parameter adjusts independently.

Adam W, proposed by Loshchilov and Hutter in twenty seventeen, modifies Adam by decoupling weight decay from the adaptive update. Standard L two regularization adds a penalty term to the loss, which means the regularization gradient goes through Adam's moment normalization. This is undesirable because the regularization strength ends up depending on the gradient history in complex ways. Adam W instead applies weight decay directly to the parameters after the Adam step, bypassing the moment estimates entirely. This gives cleaner regularization behavior.

For years, Adam W was essentially unquestioned for transformer training. People tried alternatives. Adafactor, which uses a factored second-moment estimate to save memory. L A M B, which normalizes updates per layer for large-batch training. S G D with momentum, which still works well for vision models. But for language model pretraining, Adam W was the universal default. The community accumulated years of wisdom about learning rate schedules, warm-up durations, weight decay values, beta one and beta two settings. Nobody had a strong reason to change.

Then in twenty twenty four, the modded-nanogpt project changed the conversation.

modded-nanogpt is a competitive speedrun project, led primarily by Keller Jordan and other community contributors, that builds on Karpathy's nanoGPT codebase. The goal is simple and well-defined. Train a G P T two class model, one hundred and twenty-four million parameters, to a target validation loss on the OpenWebText dataset, as fast as possible in wall-clock time on a single G P U. The leaderboard tracks who reaches the target first. Dozens of contributors optimized every part of the pipeline. Architecture modifications, learning rate schedules, data loading tricks, mixed-precision improvements, batch-size tuning, initialization schemes. Every knob was turned.

And the single largest speedup on the leaderboard came from the optimizer. Not from any architecture change. Not from data engineering. From replacing Adam W with Muon on the hidden-layer weight matrices.

Muon was proposed by Kosson, Jordan, and Bielik in twenty twenty four. The name stands for Momentum plus Orthogonalization. Let me explain the core idea carefully, because the intuition is simple once you see it.

Consider a weight matrix in a transformer. Say a four-by-four matrix, to keep the example small. This matrix maps one vector space to another. It is a linear transformation. During training, you compute the gradient of the loss with respect to this matrix. The gradient is also a matrix, the same shape as the weight. Each element of the gradient tells you how the loss changes if you slightly change the corresponding weight element.

Adam W treats this matrix as a flat collection of sixteen independent numbers. It computes a separate learning rate for each of the sixteen elements. It does not know or care that these sixteen numbers form a matrix representing a linear transformation. It processes them element-wise.

But a matrix has structure beyond its individual elements. It has singular values. It has a rank. It has a spectral norm, the largest singular value. When you think of the gradient as a matrix rather than a bag of numbers, its singular values tell you something important. They tell you the magnitude of the gradient signal along different directions in the output space. A gradient with one very large singular value and many small ones means the loss is very sensitive to changes in one particular direction and much less sensitive in others.

Adam W's element-wise update reflects the element-wise structure of the gradient, but it scrambles the matrix structure. The resulting weight update has singular values that depend on the element-wise second moments in complicated ways. The directions that receive the strongest gradient signal do not necessarily get the largest updates, because the per-element normalization redistributes magnitude.

Muon's idea is to preserve the directional information in the gradient but equalize the magnitudes. It takes the gradient matrix and orthogonalizes it using Newton-Schulz iteration, which is an iterative algorithm that converges to the closest matrix with all singular values equal to one. The resulting update direction is an orthogonal matrix. Every direction in the gradient gets equal step size. No direction is privileged over another by magnitude.

Let me describe the Newton-Schulz iteration concretely, because it is simple. Start with the gradient matrix G, normalized by its Frobenius norm so it has unit norm. Then iterate the following update five times. X becomes three-halves times X minus one-half times X times X transpose times X. Each iteration involves a few matrix multiplications. After five iterations, the matrix X has converged to a matrix whose singular values are all very close to one. This matrix is the orthogonalized gradient, and it becomes the update direction.

The full Muon update is S G D with momentum, using this orthogonalized gradient as the gradient signal. First, compute the ordinary gradient. Second, add momentum, an exponential moving average of past gradients, same as standard S G D with momentum. Third, orthogonalize the combined gradient plus momentum using Newton-Schulz. Fourth, step in the orthogonalized direction with a fixed learning rate, plus weight decay.

Why is equalizing the singular values beneficial? The geometric argument is about the norm you use to measure step size. Standard gradient descent, including Adam, measures step size in the Frobenius norm, which is the square root of the sum of squared elements. This treats the matrix as a flat vector. But for a matrix that represents a linear transformation, the natural measure of its "size" is the spectral norm, the largest singular value. Two matrices with the same Frobenius norm can have very different spectral norms, meaning they have very different effects as linear transformations.

Muon's orthogonalized update has spectral norm exactly one, regardless of the gradient's structure. This means the step in the weight space is measured by its effect as a linear transformation, not by its element-wise magnitude. For parameters that are inherently linear transformations, this is the right metric.

A critical detail. Muon is only applied to hidden-layer weight matrices. Embeddings stay on Adam W. The language model head stays on Adam W. Normalization parameters stay on Adam W. Biases stay on Adam W. The reason is that the spectral-norm argument only applies to parameters that represent linear maps between high-dimensional spaces. The embedding layer is a lookup table, not a linear map in the same sense. The normalization parameters are scalar per-channel values. These have very different geometry, and the orthogonalization argument does not apply.

In nanochat, Karpathy's full L L M training pipeline, this is implemented cleanly. The optimizer partitions parameters into two groups. Matrix-shaped hidden-layer weights get Muon. Everything else gets Adam W. Both groups can use different learning rates and weight decay values. The implementation is about ten to fifteen lines of new code on top of a standard optimizer.

The results on the modded-nanogpt leaderboard were striking. With Muon on hidden-layer weights and Adam W on everything else, the wall-clock time to reach the target loss dropped by a factor of roughly two to three. Same hardware, same architecture, same data. Just a better optimizer for the matrix-valued parameters. This is a very large free lunch.

Why was Adam W the default for so long if Muon is this much better? Several factors.

The community had accumulated extensive practical knowledge about Adam W. Learning rate schedules, warm-up strategies, the effect of beta two on training stability, how weight decay interacts with model size. Switching optimizers means losing all that institutional knowledge and re-tuning from scratch. The switching cost is real, even if the destination is better.

Optimizer research fell out of fashion after Adam. From roughly twenty fifteen to twenty twenty three, new optimizer papers rarely showed more than single-digit percentage improvements, and the improvements often did not replicate across tasks and scales. The community rationally concluded that the optimizer was not the lever to pull. Attention went to architectures, data, and scaling. The idea that a different optimizer could give a two to three times speedup was simply not in people's priors.

And the spectral-norm insight requires thinking about parameters as matrices rather than flat vectors. This perspective is natural in the mathematical optimization literature, in work on natural gradients and Riemannian optimization, but it was not standard in the deep learning practitioner community, which tends to think of parameters as lists of numbers.

Let me put Muon in the broader context of this series. In episode two, we talked about scaling laws. Loss is a smooth function of compute, data, and parameters. A better optimizer shifts the scaling curve. At the same compute budget, Muon reaches a lower loss, or equivalently, it reaches the same loss in less time. This is what "free lunch" means here. You get better results without more hardware, more data, or a different architecture. You just spend the same compute more wisely.

The Muon result is also a reminder that the transformer training stack is not fully optimized. There are real gains hiding in components that the community assumed were settled. Initialization, normalization schemes, the order of operations within a transformer block, the learning rate schedule, the way residual connections are weighted. The modded-nanogpt project found multiple such gains beyond Muon, and they stack. Architecture tweaks, better initialization, muon, and learning rate improvements together produced a speedup much larger than any individual contribution.

Where does Muon go from here? The initial results are on small models, one hundred and twenty-four million parameters. Scaling to larger models requires verifying that the Newton-Schulz iteration remains cheap relative to the backward pass, and the early indications are positive because the iteration is a fixed number of matrix multiplies per layer, which parallelizes well on G P Us. Hyperparameter transfer from small to large scale also needs validation. nanochat uses Muon by default for its larger runs, and the results look good, but hundred-billion-parameter-scale validation is still in progress as of twenty twenty five.

The optimizer is not the only part of the training loop that the modded-nanogpt project showed was underoptimized. Learning rate schedules also saw significant revision. The standard recipe for years was cosine annealing. Start with a linear warm-up from zero to the peak learning rate over the first few thousand steps. Then follow a cosine curve that smoothly decays the learning rate to near zero by the end of training. This schedule is well-tested and widely used, but it has a practical problem. You have to know the total number of training steps in advance, because the cosine schedule depends on it. If you want to train longer than planned, or if you want to reuse a checkpoint from the middle of training, the schedule does not adapt.

The alternative that gained traction in twenty twenty four is called warmup-stable-decay, or W S D. Warm up to the peak learning rate as before. Then hold the learning rate constant for most of training. Then, in the final phase, decay it quickly, linearly or with a cosine schedule, over a short window. This has two advantages. First, you do not need to commit to a total training duration upfront. You can decide when to start the decay phase based on how training looks. Second, the stable phase maintains a high learning rate throughout the bulk of training, which some researchers argue keeps the model in a better region of the loss landscape for longer.

W S D was used in the Llama three training runs and in several modded-nanogpt entries. The improvement over cosine annealing is not as dramatic as Muon's, but it is consistent, and the operational flexibility of not committing to a fixed schedule length is valuable in practice.

There is also a growing family of related optimizers. Sophia, proposed by Liu and colleagues in twenty twenty three, uses diagonal Hessian estimates for per-parameter scaling, a different way to incorporate second-order information. S O A P, by Vyas and colleagues, draws connections to the Shampoo optimizer, which uses a Kronecker-factored preconditioner that also respects matrix structure. The common thread is moving beyond element-wise adaptive methods to methods that understand the geometry of the parameter space. This was studied theoretically for years under the banner of natural gradient methods and Riemannian optimization, but making these methods practical for large-scale training with minimal overhead is what twenty twenty four contributed.

Let me close the series with a broader observation. We started in episode one with why attention replaced recurrence. Parallel computation, global information flow, no rigid inductive bias. We went through the pretraining recipe and scaling laws. Position embeddings. Inference efficiency. Mixture of experts. The post-training arc from R L H F through D P O to G R P O. Reasoning models and test-time compute. Long context. Agents. And now, optimizers.

The thread through all sixteen episodes is the same. At each stage, the field found a bottleneck, something that limited how smart or how efficient or how practical models could be, and removed it or pushed it further out. Recurrence was a compute bottleneck, and attention removed it. Data allocation was a scaling bottleneck, and Chinchilla and Llama showed how to spend it better. Absolute position encodings were a context-length bottleneck, and R o P E pushed it out. The K V cache was a serving bottleneck, and G Q A shrank it. Dense computation was a capacity bottleneck, and M o E decoupled size from cost. The R L H F pipeline was a complexity bottleneck, and D P O and G R P O simplified it. Single-pass inference was a quality ceiling, and reasoning models added test-time compute. And element-wise optimization was a training-efficiency bottleneck, and Muon showed a better way.

The pattern is not that old approaches were wrong. They were correct given their constraints. The pattern is that constraints change as the field scales, and yesterday's good-enough solution becomes today's bottleneck. The fun part is that there are always more bottlenecks to find.

Thanks for listening, and thanks for sticking with the series. I hope it has been useful.
