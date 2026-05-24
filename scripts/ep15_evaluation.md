---
episode: 15
slug: evaluation
title: "Evaluation: how we measure what models can do"
core_question: "Benchmarks, contamination, custom domain evals, chatbot arena, and why eval is the hardest unsolved problem"
target_minutes: 17
voice: en-US-AndrewMultilingualNeural
---

# Episode 15: Evaluation, how we measure what models can do

Welcome to episode fifteen. We have spent most of this series on how modern models are built, adapted, post-trained, deployed as systems, and fed with new data. But there is a question we have been dodging. How do you know if a model is actually good? Not good in the abstract. Good at the specific things you care about. Good enough to deploy. Better than the model it is replacing. This sounds like it should be straightforward. It is not. Evaluation is arguably the hardest unsolved problem in modern L L M development.

Let me start with the standard approach. Benchmarks. A benchmark is a fixed set of questions with known correct answers. You run the model on all the questions, compute the fraction it gets right, and report the score. Simple, reproducible, comparable across models.

The landscape of L L M benchmarks is large. Let me walk through the ones that matter most.

M M L U, Massive Multitask Language Understanding, is a collection of fifty-seven subjects ranging from elementary math to professional law to clinical medicine. Each question is multiple choice with four options. It was designed to measure broad knowledge and reasoning. When it was released in twenty twenty, a score of forty-three percent was state of the art. By twenty twenty five, frontier models score above ninety percent. M M L U saturated, meaning the best models are near perfect on it, which limits its ability to distinguish between top models.

G S M eight K is a set of eight thousand and five hundred grade-school math word problems. Each requires multi-step arithmetic reasoning. The answer is a specific number, making scoring unambiguous. G S M eight K became the standard benchmark for reasoning because it is easy for humans, almost all problems are solvable by a competent middle-school student, but hard for models that cannot chain logical steps. As we discussed in episodes nine and ten, reasoning models dramatically improved G S M eight K scores, and the benchmark is now close to saturated for frontier models.

HumanEval is a code generation benchmark. One hundred and sixty-four Python programming problems. The model generates a function body, and the output is tested against unit tests. The metric is pass at k, the fraction of problems where at least one of k generated solutions passes all tests. HumanEval measures functional code generation, the ability to produce code that actually runs correctly, not just code that looks plausible.

MATH is a harder math benchmark, with problems from high-school and competition-level mathematics. Unlike G S M eight K, which is arithmetic reasoning, MATH includes algebra, geometry, number theory, and calculus. Scores on MATH were in the single digits when it was released. Frontier reasoning models now score above eighty percent on some subsets.

S W E-bench, Software Engineering Bench, is a benchmark for coding agents. Each problem is a real GitHub issue from a real open-source repository. The model, or agent, must read the issue, navigate the codebase, identify the relevant files, make the correct code changes, and produce a patch that passes the repository's test suite. S W E-bench is much harder than HumanEval because it requires understanding large codebases, not just writing isolated functions. It is the benchmark that evaluates the kind of coding agent that tools like Claude Code and Cursor implement. As of early twenty twenty five, the best agents solve around fifty percent of S W E-bench verified problems, up from less than five percent a year earlier.

These benchmarks share a common structure. Fixed questions, known answers, automatic scoring. This makes them reproducible and comparable. But it also makes them vulnerable to a problem that has become the central crisis of L L M evaluation. Contamination.

Contamination means the model has seen the benchmark data during training. If the model was trained on a web scrape that included the benchmark questions and answers, it might be recalling answers rather than reasoning about them. Its benchmark score overstates its actual capability on novel problems of the same type.

This is not hypothetical. Benchmarks like M M L U and G S M eight K have been online for years. Their questions appear on websites, in blog posts, in discussions. Any large-scale web scrape is likely to contain at least some benchmark questions. And the problem is insidious, because contamination can be partial. The model might have seen a similar question, or a paraphrase, or a discussion of the answer, without having seen the exact question. Detecting this is extremely difficult.

The response to contamination has been threefold. First, held-out benchmarks. Create new benchmarks that are not public, run models against them in controlled settings, and never release the questions. This works but limits reproducibility. If only one lab has the benchmark, other labs cannot verify the results.

Second, dynamic benchmarks. Generate new questions procedurally, so the specific questions have never been seen before. Live Bench and other dynamic evaluation suites take this approach. The tradeoff is that procedurally generated questions may not capture the same difficulty distribution as curated human-written questions.

Third, human evaluation. Have humans interact with the model and judge the quality of its responses. This is immune to contamination because the interactions are novel. But it is expensive, slow, and subjective.

The most successful human evaluation system is Chatbot Arena, run by L M S Y S at U C Berkeley. The setup is elegant. A user submits a prompt. Two anonymous models generate responses. The user picks which response they prefer, without knowing which model is which. The preferences are aggregated using an E L O rating system, the same system used to rank chess players. Each model gets a rating that reflects how often it wins head-to-head comparisons.

Chatbot Arena has become the most trusted single measure of overall model quality, because it is hard to game. The prompts come from real users with diverse needs, not from a fixed set that could be trained on. The evaluation is comparative, so absolute quality does not matter, only which model is better. And the E L O system produces a ranking that is remarkably stable and consistent with expert judgments.

But Chatbot Arena also has limitations. The user base is skewed toward technical users who ask coding and reasoning questions. Models that are good at coding tend to rank disproportionately well. Conversational quality, safety, nuance, and calibration are underweighted because those are harder for casual users to evaluate in a quick side-by-side comparison. And the rating is a single scalar, which collapses many dimensions of quality into one number. A model that is excellent at code and mediocre at creative writing might have the same E L O as a model that is mediocre at code and excellent at creative writing.

There is a deeper philosophical issue with evaluation that is worth stating directly. Benchmarks measure what you can test. But the most valuable capabilities of modern L L Ms are often the ones that are hardest to test. Following complex multi-constraint instructions. Maintaining a coherent persona over a long conversation. Knowing when to say "I don't know." Calibrating confidence appropriately. Being helpful without being sycophantic. These are real, important capabilities, and none of them have good automated benchmarks.

This leads to what practitioners call the "vibes versus metrics" tension. A model can score higher on every benchmark and still feel worse to use, because the benchmarks do not capture the qualities that matter for the specific use case. Conversely, a model can score lower on benchmarks but be preferred by users because it handles the kinds of queries those users actually make. The gap between benchmark performance and user satisfaction is real, and it is why model selection in production is often based on qualitative evaluation alongside quantitative scores.

Let me also touch on evaluation for safety, because this is a distinct and important area. Red teaming is the practice of adversarially testing a model by trying to elicit harmful, biased, or dangerous outputs. This is typically done by human testers who try creative attack strategies, jailbreaks, social engineering, obfuscation. Automated red teaming uses another L L M to generate adversarial prompts at scale, which is faster but may miss novel attack vectors that a creative human would find.

Safety benchmarks like Truthful Q A measure calibration and honesty. Toxi Gen measures toxic language generation. B B Q measures social biases. These provide useful signal but are narrow. The real safety evaluation for a deployed model is ongoing monitoring of production traffic, flagging responses that violate safety policies, and continuously updating the model's training data and R L H F signal to address emerging issues.

Let me also mention the evaluation tools that practitioners use day to day. The L M eval harness, originally from EleutherA I, is an open-source framework that runs a model against dozens of standard benchmarks with consistent formatting and scoring. It is the closest thing to a standard evaluation suite, and most model release papers report L M eval harness numbers. H E L M, from Stanford, provides a broader evaluation framework that includes not just accuracy but also calibration, robustness, fairness, and efficiency. For agent evaluation, the S W E-bench harness provides the infrastructure to run coding agents against real GitHub issues and verify their patches.

Let me close with two concrete examples that illustrate these evaluation challenges.

First, contamination detection. In twenty twenty three, researchers at Scale A I tested G P T four on a benchmark that had been released after G P T four's training cutoff, and compared its performance to benchmarks that existed before the cutoff. G P T four scored significantly higher on pre-cutoff benchmarks than on post-cutoff ones of similar difficulty. This is circumstantial evidence of contamination. Not proof, because many factors differ between benchmarks, but suggestive. A cleaner test is n-gram overlap. Check whether benchmark questions appear verbatim, or nearly verbatim, in known training data. The Llama three technical report includes a contamination analysis that flags benchmarks with high n-gram overlap and reports results both with and without contaminated subsets. This should be standard practice in model releases, but it is not yet universal.

Second, designing domain-specific evaluation. Suppose you are deploying a model for medical question answering. Standard benchmarks like Med Q A exist, but they are multiple-choice questions about medical knowledge, which does not capture what doctors actually need. A better evaluation might present the model with a clinical case, a patient history and symptoms, and ask it to generate a differential diagnosis with a treatment plan. You would have a panel of physicians score the outputs on accuracy, completeness, and safety. This is expensive, slow, and not scalable. But it is the only evaluation that tells you whether the model is ready to deploy. The gap between standard benchmarks and domain-specific evaluation is where most deployment failures hide.

The hard truth about evaluation is that no single metric, benchmark, or methodology captures what we care about. Benchmarks provide reproducible scores but saturate quickly and are vulnerable to contamination. Human evaluation through systems like Chatbot Arena gives the most trusted overall signal but is expensive and collapses many dimensions into one. The gap between benchmark performance and real-world usefulness is real and persistent. The best evaluation practice combines automated benchmarks for regression testing, human evaluation for overall quality, and domain-specific testing for the particular use case, and accepts that no single metric captures what we really care about.

In the final episode, we are going to come back inside the model one last time. We are going to talk about optimizers, the component of the training stack that everyone assumed was solved. And how a new optimizer called Muon showed that there was a large free lunch hiding in how we update weight matrices.

Thanks for listening. See you next time.
