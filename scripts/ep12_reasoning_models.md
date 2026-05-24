---
episode: 12
slug: reasoning_models
title: "Reasoning models and test-time compute"
core_question: "o1, R1, the s1 finding that 1k traces is near-frontier reasoning"
target_minutes: 22
voice: en-US-AndrewMultilingualNeural
---

# Episode 12: Reasoning models and test-time compute

Welcome to episode twelve. This episode is about a fundamental shift in how we think about making models smarter. For the first seven years of the transformer era, from twenty seventeen through twenty twenty three, the playbook was clear. Want a smarter model? Make it bigger. Train it on more data. Spend more compute at training time. Scaling laws, as we covered in episode two, told you exactly what you would get. The entire field optimized along two axes. Model size and training data.

Starting in twenty twenty four, a third axis became headline-worthy. Test-time compute. Instead of making the model bigger, let it think longer. Let it spend more tokens of internal reasoning before producing an answer. The finding was that accuracy on hard reasoning tasks improves predictably with thinking time, analogous to how it improves with training compute, but along a completely different and independent dimension.

This is the story of reasoning models. OpenAI's o one, released in September twenty twenty four. DeepSeek's R one, released in January twenty twenty five. And the surprising finding from the s one paper that you might not even need R L to get most of the way there.

Let me start with what "test-time compute" means concretely, because the term is used loosely and it helps to be precise. A standard language model, when you ask it a question, generates its answer in a single pass. One token after another, left to right, until it produces a stopping token. The total compute is proportional to the number of output tokens times the cost per token, which is determined by the model size. For a short answer, a few hundred tokens. For a long answer, a few thousand. The model does not "think" in any meaningful sense. It maps input to output through one forward pass per token.

A reasoning model does something different. Before producing its final answer, it generates an extended internal chain of thought. This chain of thought is additional text, produced token by token just like any other output, but it serves as a scratchpad for working through the problem. The model might spend five thousand tokens working through a math problem, trying one approach, realizing it does not work, backtracking, trying another approach, verifying intermediate results, and eventually arriving at a final answer that it then states in one sentence. The total compute is much higher, not because the model is bigger or the per-token cost is different, but because it produces many more tokens.

The key empirical finding is that this additional compute is productive. On tasks like competition math, formal logic, and complex code generation, accuracy increases as you let the model think longer. The relationship follows a roughly predictable curve. Plot accuracy against the log of the number of thinking tokens, and you get something that looks like a smooth scaling law, analogous to the training-compute scaling laws from episode two, but in a different dimension.

OpenAI demonstrated this first with o one. The exact training methodology of o one is not public, and OpenAI does not expose the full chain of thought. In the public interface, you see a summary of the thinking process, not the raw trace. But the summaries and the model's observable behavior strongly suggest extended internal reasoning that can run to thousands of tokens.

The observable patterns suggest several distinctive behaviors. The model explores multiple approaches to a problem. It computes intermediate results and checks them. It explicitly states when an approach is not working and switches to another one. It verifies its final answer by re-deriving it through a different method. These behaviors are reminiscent of how a careful human mathematician works through a difficult problem, not just computing but also monitoring, backtracking, and verifying.

o one's results on competition math were the headline. It scored ninety-three percent on the A I M E, the American Invitational Mathematics Examination, compared to around thirteen percent for G P T four. On the Codeforces competitive programming platform, o one reached the eighty-ninth percentile, compared to the eleventh percentile for G P T four. These are not incremental improvements. They are qualitative jumps. And they came not from a bigger model or better pretraining data, but from a model that was trained to think before answering.

DeepSeek R one provided the first detailed, public account of how to train such a model. As we covered in the last episode, the recipe uses G R P O with verifiable rewards. But the reasoning story goes beyond the R L algorithm itself. Let me describe the R one training pipeline in more detail.

Stage one is cold start S F T. The DeepSeek team collected a few thousand high-quality reasoning traces, examples of detailed, step-by-step problem solving on math and code tasks. These were produced by prompting a capable model with instructions to show its work. The base model is fine-tuned on these traces using standard supervised fine-tuning. This stage does not teach the model to reason. It teaches the model the format. Write your thinking inside thinking tags. Show your steps. State your final answer clearly at the end.

Stage two is G R P O training on verifiable tasks. This is where the actual reasoning improvement happens. The model samples multiple completions for each math or code problem, gets a binary reward for correctness, and the policy gradient pushes the model toward patterns that lead to correct answers. As we discussed in episode eleven, the model develops emergent reasoning behaviors during this phase. Self-correction. Backtracking. Exploration of alternative approaches.

Stage three is rejection sampling to create a high-quality S F T dataset for general tasks. The R one model from stage two is very good at reasoning but may have regressed on general instruction following. The team uses the reasoning model to generate responses to a broad set of prompts, filters for quality using automated evaluators, and creates a new S F T dataset that combines reasoning capability with general helpfulness. The model is then fine-tuned on this mixed dataset.

Stage four is a final round of R L, now using a mix of verifiable rewards for reasoning tasks and a reward model for general tasks. This final stage polishes the model's overall behavior.

What makes R one's training story technically interesting is what emerges during the G R P O phase. The model develops specific reasoning strategies without being explicitly trained on examples of those strategies.

Self-verification. The model solves a problem, then re-derives the answer using a different method to check consistency. For example, solving an equation algebraically and then verifying by substituting the answer back into the original equation.

Backtracking. The model starts down one approach, computes for a while, realizes the approach is not leading anywhere or has produced an error, and explicitly restarts with a different strategy. The transition is often marked by phrases like "wait, this does not seem right" or "let me try another way."

Decomposition. The model breaks complex problems into subproblems, solves each independently, and then combines the results. This is especially evident on multi-step word problems.

Extended exploration. On very hard problems, the model generates thousands of tokens of reasoning, trying many different approaches, before finding one that works. It does not give up after the first attempt.

These behaviors emerged because the reward gradient, which only depends on whether the final answer is correct, propagates back through every token of the reasoning chain. Token patterns that preceded correct answers get reinforced. Token patterns that preceded incorrect answers get weakened. The model learns that writing "let me check this by..." before a verification step is instrumentally useful, because it leads to self-correction that leads to more correct final answers. The reasoning style is a means to an end, discovered by the optimization process.

Now here is where the story takes a surprising turn. The s one paper, from a group at Stanford including Muennighoff and colleagues, published in early twenty twenty five, asked whether R L is actually necessary for reasoning behavior.

Their experiment was clean. They took Qwen two point five, a strong thirty-two billion parameter base model. They collected about one thousand high-quality reasoning traces from frontier reasoning models like o one and R one. These traces showed the extended thinking style, with explicit step-by-step derivation, self-correction, and careful verification. They fine-tuned the base model on just those one thousand examples, using standard supervised fine-tuning. No R L. No reward model. No G R P O. Just S F T on a thousand examples of good reasoning.

The result was startling. The S F T only model, which they called s one, achieved performance close to o one mini on several math benchmarks. Not o one, which had extensive R L training, but o one mini, which is itself a very strong reasoning model. And it achieved this with just S F T on one thousand examples.

The implication is that reasoning capability is mostly latent in large pretrained models. The pretraining phase, which involves predicting the next token across trillions of tokens of text that includes mathematical derivations, code, logical arguments, and step-by-step explanations, gives the model the underlying ability to chain logical steps together. What is missing is not the capability but the format. The model does not know it should think step by step. S F T on reasoning traces teaches the format, and the latent capability does the rest.

This does not mean R L is useless. R one with G R P O achieves substantially higher performance than s one on the hardest problems. R L helps the model discover novel reasoning strategies not present in the S F T data. It helps the model generalize to problem types it has not seen examples for. And it allows the model to improve beyond the quality ceiling of the S F T data, because it is optimizing against an objective, not imitating examples. But the s one result shows that the bulk of the reasoning gains come from elicitation, not construction. S F T gets you most of the way. R L gets you the rest.

The s one paper also introduced a practical and elegant technique called budget forcing, which gives you a controllable knob over test-time compute.

Here is how it works. During generation, the model produces its thinking phase inside special tokens, and eventually tries to end the thinking by generating an end-of-thinking token. Budget forcing intercepts this. If you want more thinking, you suppress the end token and instead append the text "Wait," to the model's context. The model, seeing "Wait," in its own output, interprets this as a prompt to continue thinking. It might reconsider its approach, double-check a step, or explore an alternative. You can do this multiple times, each time injecting "Wait," to extend the thinking phase.

If you want less thinking, you truncate the thinking phase at a specified number of tokens and force the model to begin its final answer immediately.

The result is a smooth, controllable tradeoff between compute and accuracy. More thinking tokens generally yield higher accuracy, up to a saturation point where additional thinking stops helping or even hurts, because the model starts second-guessing correct answers. Less thinking yields lower accuracy but faster and cheaper responses.

The shape of this tradeoff curve is informative. For easy problems, even a few hundred thinking tokens suffice, and additional thinking adds no value. For hard problems, accuracy keeps climbing with more thinking tokens up to several thousand. For very hard problems, even ten thousand tokens of thinking may not be enough. The optimal thinking budget is problem-dependent, which suggests that a production system should dynamically allocate thinking time based on estimated problem difficulty.

Let me make the analogy to training scaling laws explicit, because this conceptual connection is the key takeaway. In episode two, we covered the observation that loss is a smooth power-law function of training compute. More training compute, lower loss, predictably. The test-time compute result is directly analogous. Performance is a smooth, roughly predictable function of inference compute, measured in thinking tokens. More thinking, better performance, with diminishing returns.

These two axes are largely independent. You can improve a model by spending more at training time, bigger model, more data. Or you can improve results by spending more at inference time, more thinking tokens. Or both. They compose. A larger model that also thinks longer will outperform either a larger model that answers immediately or a smaller model that thinks longer.

This independence has profound practical implications. Training compute is a one-time fixed cost. You spend it once and serve the resulting model for months or years. Test-time compute is a per-request variable cost. Each query costs more if you let the model think longer. This means you can dynamically allocate intelligence. Easy questions get fast, cheap, direct answers. Hard questions get slow, expensive, deeply reasoned answers. A routing layer, which could itself be an L L M classifier, estimates the difficulty of each query and decides how much thinking budget to allocate. The average cost per query drops substantially compared to giving every query the maximum thinking budget.

Let me survey the broader landscape of test-time compute techniques, because chain-of-thought reasoning is not the only approach.

Best of N sampling. Generate N independent completions and pick the best one. With a verifiable reward, you check which completions are correct and return one of the correct ones. Without a verifiable reward, you can use majority vote, returning the answer that appears most frequently across the N completions. This is a brute-force approach to test-time compute. It is simple, embarrassingly parallel, and effective. At N equals sixty-four, majority vote on G S M eight K can close most of the gap between a base model and a reasoning model, without any special training.

Self-consistency, proposed by Wang and colleagues in twenty twenty three, is a refined version of majority vote specifically for chain-of-thought reasoning. Generate multiple independent reasoning chains, extract the final answer from each, and take the majority vote. The insight is that correct reasoning chains tend to converge on the same answer through different paths, while incorrect chains tend to scatter across different wrong answers. Self-consistency is surprisingly effective and composes with other techniques.

Tree search. Instead of generating a single linear chain of thought, explore a tree of possible reasoning paths. At each step, generate multiple candidate continuations, evaluate them using a value model or heuristic, and pursue the most promising branches. Prune unpromising branches early. This is closer to how classical A I search works, think game tree search in chess, and it can find solutions that a single linear chain misses by exploring more of the reasoning space. Monte Carlo tree search applied to L L M reasoning is an active research area in twenty twenty five.

Iterative refinement. Generate an initial answer, then feed it back to the model with a prompt asking it to critique and improve. Repeat for several rounds. Each iteration uses additional compute and typically produces a better answer, though with diminishing returns. This is related to the reflection pattern we will discuss in episode thirteen.

All of these share the same principle. Trade more inference compute for better results. The specific mechanism differs, but the empirical scaling behavior is similar. Performance improves smoothly and predictably with additional compute, up to a task-dependent saturation point.

There is one more result from the R one paper that has large practical implications. Distillation. The DeepSeek team showed that you can take a large reasoning model like R one, generate reasoning traces from it on a diverse set of problems, and use those traces as S F T data to train much smaller models. The resulting models, called R one Distill, range from one point five billion to seventy billion parameters. The remarkable finding is that the distilled models outperform models of the same size that were trained with R L directly. A seven billion parameter model distilled from R one's traces beats a seven billion parameter model trained with G R P O from scratch on the same tasks.

Why does distillation work so well? Because the reasoning traces from R one are higher quality than what a small model would discover through its own R L exploration. The small model's R L process is limited by its own capacity to explore the space of reasoning strategies. The large model has already found effective strategies, and the small model can imitate them even if it could not have discovered them independently. This is analogous to a student learning calculus techniques from worked examples rather than re-deriving them from first principles.

The practical implication is significant. You do not need to run expensive R L on every model you deploy. You can run R L on one large model, collect its reasoning traces, and distill those traces into a family of smaller models for different deployment scenarios. The large model pays the exploration cost once. The small models inherit the results cheaply.

There is also the question of when more thinking actually hurts. On easy problems, where the model's first instinct is correct, extended thinking can be counterproductive. The model solves the problem correctly in the first few hundred tokens, then keeps thinking because the format encourages it, and in the process second-guesses itself into a wrong answer. This is the overthinking problem. Budget forcing helps, by letting you set a low thinking budget for easy problems. But knowing which problems are easy in advance is itself a hard classification problem.

Let me close with the broader limitations. Test-time compute scaling works best on tasks with clear, verifiable correctness criteria. Math, code, formal logic, factual questions with known answers. For open-ended tasks like writing an essay, giving advice, or summarizing a nuanced document, more thinking does not always help. The model might overthink, second-guess itself, produce a wandering response, or fixate on details that do not matter. Knowing when to think deeply and when to just answer directly is itself a skill, and current reasoning models do not always get this right.

There is also a cost consideration that is easy to underestimate. A reasoning model that spends ten thousand tokens thinking about every question is roughly ten times more expensive per query than a model that answers in a thousand tokens. For many applications, the direct answer from a strong base model is good enough, and the extra cost of reasoning is not justified. The value of reasoning models is concentrated on the hard tail of queries.

Finally, there is the question of whether thinking tokens are the most efficient way to spend additional inference compute. Generating tokens sequentially is inherently slow, bounded by the model's autoregressive generation speed. Best-of-N sampling can be parallelized across multiple G P Us, potentially delivering more "thinking" per wall-clock second. The optimal allocation between depth of thinking, which is sequential, and breadth of sampling, which is parallelizable, depends on the task, the hardware, and the latency requirements. This is an active area of research and engineering.

If I had to compress this episode. From twenty seventeen through twenty twenty three, the way to make models smarter was to scale training. Starting in twenty twenty four, test-time compute became a second scaling axis. Let the model think longer and it gets smarter, predictably. o one and R one demonstrated this with extensive R L training. s one showed that much of the capability can be unlocked with just one thousand S F T examples, no R L required. Budget forcing gives a controllable knob over the accuracy-cost tradeoff. The result is a new paradigm where intelligence is dynamically allocated at inference time.

In the next episode, we are going to put models into the real world. We will talk about agents, the pattern of giving a language model tools and letting it take actions in a loop. ReAct, planner-executor, standardized tool calling, and the hard-won lesson that simple workflows almost always beat complex multi-agent systems.

Thanks for listening. See you next time.
