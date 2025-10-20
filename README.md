# AI research — notes and diary

This repository is a running research diary: notes, experiment notebooks, short essays, and small utilities I use while exploring machine learning, alignment, and large models. The contents are curated and maintained personally — I am not accepting outside contributions.

Contents are pragmatic and evidence-oriented: reproducible snippets, pointers to papers, and working notebooks. See the top-level folders for organized material (for example `LangChain/`, `Finetuning/`, `Embeddings/`).

## Intent and signal

* Purpose: a personal research diary and reference index.
* Tone: technical and critical where appropriate; concise where possible.
* Availability: curated privately; do not submit pull requests or expect formal review.

## Short technical history (n-grams → Transformers)

This is a compact timeline to explain why the 2017 Transformer paper is a structural turning point.

* N-gram and statistical language models — count-based approaches with explicit smoothing and transparent failure modes; limited to short contexts.
* Neural sequence models (RNNs, LSTMs) — learned representations and improved generalization, but sequential training and inference limited long-range modeling in practice.
* Attention and the Transformer (2017) — Vaswani et al.'s self-attention enabled direct token-to-token interactions, large-scale parallelism, and a flexible substrate for language and multimodal models.

Key influences: Geoffrey Hinton, Yann LeCun, Yoshua Bengio and others established principles (distributed representations, optimization approaches, and engineering practices) that made large-scale, stable training possible.

## Current focus areas

* Alignment and safety: failure modes, reward misspecification, and methods for robust oversight.
* Hallucinations and grounding: when models produce confident-but-wrong outputs and how to anchor outputs to verifiable context.
* Interpretability: mechanistic investigations, circuit-level structure, and representation analysis.
* Scaling and capability analysis: tracking behavior changes with scale while avoiding simple emergent narratives.
* World models: latent simulators and model-based approaches that support planning and long-horizon reasoning.

## Forecast: world models as a core axis

I treat world models — learned compact simulators or latent dynamics — as a central research hypothesis. Such models offer a route to richer internal representations, planning, and counterfactual reasoning. This hypothesis guides the experiments I prioritise and the literature I follow.

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

## How I use this repository

* Quick notes and bookmarks: short markdown files capturing observations.
* Notebooks: experimental trails and reproducible steps (look under `Finetuning/`, `Embeddings/`, `LangChain/`).
* Scripts and demos: small utilities used to test or reproduce ideas.

## Practicalities

* License and contributions: personal collection; not open for contributions.
* Contact and provenance: public profiles and blog links are on my website; use those channels for inquiries.
