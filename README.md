# AI research — notes and diary

This repository is a running research diary: notes, experiment notebooks, short essays, and small utilities I use while exploring machine learning, alignment, and large models. The contents are curated and maintained personally — I am not accepting outside contributions.

Contents are pragmatic and evidence-oriented: reproducible snippets, pointers to papers, and working notebooks. See the top-level folders for organized material (for example `LangChain/`, `Finetuning/`, `Embeddings/`).

## Intent and signal

* Purpose: a personal research diary and reference index.
* Tone: technical and critical where appropriate; concise where possible.
* Availability: curated privately; do not submit pull requests or expect formal review.

## Technical history: n-grams to Transformers (and why it matters)

This section traces the intellectual lineage and failure modes that shaped modern sequence modeling. I include it because understanding *why* each shift happened — what broke and what was gained — is essential to understanding current limitations.

### N-grams and statistical language modeling

Early language models (Shannon's information theory, Markov chains, Katz smoothing) estimated $P(\text{next token} | \text{context})$ by counting n-gram statistics and applying smoothing techniques (Good-Turing, Kneser-Ney, etc.). These models are interpretable: you can see which n-grams drive predictions and debug failures. But they have hard limitations:

* Context window is fixed and small (typically 2–5 tokens).
* Sparse statistics: rare n-grams get poor estimates even with smoothing.
* No learned representations; similarity between words must be encoded manually.

Despite these limits, n-gram models powered search engines and speech recognition for decades and remain useful for anomaly detection and linguistic analysis.

**Key references:** Shannon (1951), Katz (1987) on backoff smoothing, Kneser & Ney (1995) on modified counting.

### Neural approaches: RNNs, LSTMs, and the sequential bottleneck

In the 1990s and 2000s, recurrent neural networks (RNNs) and long short-term memory networks (LSTMs; Hochreiter & Schmidhuber, 1997) offered learned representations and better long-range modeling than fixed-window n-grams. The key insight: a hidden state $h_t$ summarizes history, allowing the network to carry information across long sequences.

Why it mattered: LSTMs solved the vanishing gradient problem and proved that neural sequence models could learn hierarchical, abstract representations. They enabled progress on machine translation, speech, and sequence tagging.

Why it was still limited:

* **Sequential bottleneck:** Processing token $t$ requires output from token $t-1$. This prevents parallelization during training (a major scalability issue as data and models grew).
* **Long-range dependencies still fragile:** Even LSTMs struggle to reliably attend to distant context; attention weights decay or get drowned in noise.
* **No efficient way to reweight importance across the sequence:** the hidden state is a bottleneck through which all history must pass.

**Key references:** Hochreiter & Schmidhuper (1997, LSTMs), Cho et al. (2014, GRUs), Bahdanau et al. (2015, additive attention).

### Attention and the Transformer (2017)

Bahdanau et al. (2015) introduced *attention*: a mechanism to compute a weighted sum of encoder outputs, let the decoder "look at" relevant parts of the input without a tight bottleneck. This was a crucial step, but attention was grafted onto RNNs; you still had sequential encoding.

Vaswani et al. (2017) in "Attention Is All You Need" went further: remove the recurrence entirely. Use *self-attention* (compute interactions between all tokens in parallel) and stack them into a Transformer. The core insight:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where $Q$, $K$, $V$ (queries, keys, values) are learned projections. This allows:

* **Massively parallel training:** all tokens interact in one forward pass; no sequential dependency.
* **Direct long-range connections:** any token can attend to any other token with a single hop (or a few hops in multi-head / multi-layer attention).
* **Efficient scaling:** with proper hardware (GPUs, TPUs), you can train on much larger datasets and models.

This architectural choice proved to be the right abstraction for language. When paired with large-scale pretraining (BERT, GPT), it became the substrate for modern LLMs.

**Key references:** Vaswani et al. (2017), Devlin et al. (2019, BERT), Radford et al. (2019, GPT-2).

### The connection: solving a specific bottleneck

The narrative is not "neural is better than statistical" — it is "each step removed a computational bottleneck that prevented learning on scale." N-grams were bottlenecked by context size and sparsity. RNNs removed context-size limits but hit the sequential training bottleneck. Transformers removed the sequential bottleneck, enabling massive-scale pretraining on diverse data.

The irony: each step brought new problems. LSTMs are hard to parallelize but remain interpretable. Transformers parallelize beautifully but their attention patterns and internal representations are opaque. This opacity is partly why we face "hallucinations," brittle reasoning, and alignment challenges today.

## Current focus areas

* Alignment and safety: failure modes, reward misspecification, and methods for robust oversight.
* Hallucinations and grounding: when models produce confident-but-wrong outputs and how to anchor outputs to verifiable context.
* Interpretability: mechanistic investigations, circuit-level structure, and representation analysis.
* Scaling and capability analysis: tracking behavior changes with scale while avoiding simple emergent narratives.
* World models: latent simulators and model-based approaches that support planning and long-horizon reasoning.

## Forecast: why world models, not just larger LLMs

I am skeptical that scaling language modeling alone leads to AGI or robust reasoning. Here is my view:

**The LLM ceiling:** Large language models are statistical compression engines. They excel at predicting plausible continuations of text in domains well-covered by training data. But:

* They do not plan; they do not form and test hypotheses in a mental model.
* They confabulate when asked to reason about scenarios outside their training distribution (distributional robustness is poor).
* They have no internal model of causality or consequence; they learn correlations in text and sometimes extract causal structure by accident.
* Scaling alone does not fix these; it just makes the confabulations more fluent.

**The world model hypothesis:** Instead, I believe the next frontier is *learning compact, simulable models of environments* — world models. These would support:

* **Planning:** an agent that runs forward simulation to evaluate action sequences.
* **Counterfactual reasoning:** "what if I did X instead?"
* **Transfer and generalization:** a model of physics or social dynamics learned from one domain can apply to new ones.
* **Interpretability:** a learned simulator has explicit dynamics, which can be inspected and debugged.

This is not new (Ha & Schmidhuber, 2018; Schmidhuber's work in the 1990s on curious agents). But I see it as under-explored relative to LLM scaling. Yann LeCun's 1989 work on energy-based models and his recent essays on the future of AI both point toward world models as central. That convergence is not accidental.

**Current research:** Models like V-JEPA (LeCun et al., 2023), MuZero (Schaal et al., 2020) and recent work in model-based RL show this is tractable. But the paradigm is less hyped and less well-funded than LLM scaling. I treat world models as the more interesting frontier.

**Caveat:** LLMs will likely remain useful as components (reasoning engines, written output, planning aids). The claim is that LLMs alone are insufficient; we need them *embedded in* world models, not as the top-level system.

## Inspirations and labs

A few people and organizations that shape how I think about research:

**Yann LeCun.** His 1989 work on energy-based models and convolutional networks established principles (learned hierarchies, inductive biases, efficient architectures) that underpin modern deep learning. More recently, his essays on "the future of AI" and world models are prescient: he argues convincingly that LLM-only approaches are insufficient and that learning models of the world is essential. His consistency across decades is rare and humbling.

**Dario Amodei and Anthropic.** Dario co-founded OpenAI but left to start Anthropic with a focus on interpretability and safety. Anthropic's research on mechanistic interpretability (e.g., circuits, superposition, steering vectors) is among the most rigorous and reproducible I've seen. They treat safety not as an afterthought but as a constraint that shapes architecture choices. The intellectual honesty — publishing negative results, admitting uncertainty — is rare in a field full of hype.

**Chris Olah and mechanistic interpretability.** His visual essays on neural networks (e.g., "Distill" blog) set a standard for clarity and rigor. His work on superposition and polysemantic neurons opened new questions about what's actually happening in the learned representations. Inspired Anthropic's interpretability agenda.

**Neel Nanda and Janus.** Neel's public work on mechanistic interpretability and circuit analysis (on LessWrong, Alignment Research Center) is obsessively thorough and reproducible. His focus on small, interpretable models as a testbed is pragmatic — understand GPT-2 before claiming you understand GPT-4. Janus's essays on agency, world models, and the structure of cognition are deeply original and often ahead of mainstream discourse.

**Andrej Karpathy.** His pedagogy (course on neural networks, "Neural Networks: Zero to Hero") demystifies deep learning and makes you think carefully about what you're building. His move to Tesla and then back to AI research reflects someone willing to test ideas in the real world. His skepticism about AI scaling alone (implicit in his trajectory) aligns with world-model thinking.

**DeepMind.** Known for hiring top mathematicians and scientists and giving them room to pursue long-term, risky bets. Their work spans causal reasoning (Pearl's work, causal graphs), world models (RSSM, Dreamer), and fundamental questions in RL. They treat AI as a tool for scientific discovery, not just prediction. That framing attracts people asking deeper questions.

**OpenAI.** OpenAI scaled LLMs to mainstream visibility. But their recent work on reasoning (o1 chain-of-thought) and their emphasis on capability benchmarks suggests they too are recognizing LLM limits and exploring next steps. Still, I see them as more focused on scaling and deployment than on mechanistic understanding or safety.

**Anthropic (again).** I list them twice because their focus on interpretability as a first-class research goal — not a side project — is singular. Dario's public writing on AI risk and interpretability is thoughtful and resists simplistic narratives. Their hiring and research direction suggest they're taking seriously the gap between "LLMs are impressive" and "LLMs are safe or controllable."

**My view:** the most interesting work in AI is happening where skepticism meets rigor. DeepMind and Anthropic cultivate that culture. OpenAI has the scale and resources but, I suspect, less internal critique. Yann's insistence on world models over LLM scaling is a useful counterweight to hype. Neel's focus on interpretability of *small* models (because you can actually debug them) is the right research strategy. I follow their lead.

## Key AI safety issues I track

1. Specification and reward design: mis-specified objectives lead to undesirable optimization.
2. Interpretability gaps: limited mappings from internal state to behavior impede safe intervention.
3. Robustness and distributional shift: silent failure under new conditions is a systemic risk.
4. Capability overhang and misuse: rapid capability improvements can outpace governance and understanding.
5. Scalable oversight: human feedback, automated evaluation, and robust evaluation harnesses require research effort.

## Selected reading and pointers

* Vaswani et al., "Attention Is All You Need" (2017) — the Transformer architecture.
* Work on world models (Ha & Schmidhuber) and followups in model-based representation learning.
* Recent surveys and papers on hallucinations, scaling laws, and mechanistic interpretability (see folders for local copies and notes).

For curated public material, see my website: [ashioyajotham.github.io](https://ashioyajotham.github.io/) (research, projects, and blog links are there).

First favourites in research*****************

1. [Exploiting Novel GPT-4 APIs](https://arxiv.org/abs/2312.14302)

2. [Orca: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/pdf/2306.02707)

* *Can large language models (LLMs) train themselves?*
Credits: [Cameron Wolfe](https://cameronrwolfe.substack.com/p/orca-properly-imitating-proprietary) found through this [twitter thread](https://twitter.com/cwolferesearch/status/1673398297304911872)

## How I use this repository

* Quick notes and bookmarks: short markdown files capturing observations.
* Notebooks: experimental trails and reproducible steps (look under `Finetuning/`, `Embeddings/`, `LangChain/`).
* Scripts and demos: small utilities used to test or reproduce ideas.

## Practicalities

* License and contributions: personal collection; not open for contributions.
* Contact and provenance: public profiles and blog links are on my website; use those channels for inquiries.
