# AI Research

Welcome to my AI diary ^_^. Here I will be documenting interesting research papers, articles, and resources that I come across in the field of artificial intelligence, particularly focusing on large language models (LLMs) and transformers.

## What are transformers?

* `Transformers` are a type of neural network architecture that allow for parallelization across the sequence. This means that the network can process all of the tokens in the sequence at the same time, rather than having to process them sequentially. This is a huge advantage over RNNs, which must process tokens sequentially.
* It was introduced through the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) in 2017 which can be found in the Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS 2017). by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin
* Below is a diagram of the Transformer architecture:
![Transformer Architecture](
https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png
)
* Sebastian Ratchka sums it well in [Sebastian Ratchka's LinkedIn post](https://www.linkedin.com/posts/sebastianraschka_ai-llm-transformers-activity-7074387165543092224-tlX-?utm_source=share&utm_medium=member_desktop)

****************

****************

Before the transformer architecture, RNNs and LSTMs were the go-to architectures for sequence modeling tasks. However, they struggled with long-range dependencies due to their sequential nature. Transformers, on the other hand, utilize self-attention mechanisms that allow them to weigh the importance of different tokens in a sequence, regardless of their position. This enables them to capture long-range dependencies more effectively.
N-gram models existed before transformers as well, but they were limited in their ability to capture long-range dependencies and context. Transformers, with their self-attention mechanisms, have largely supplanted n-gram models in many applications due to their superior performance.
Look up the [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) for a step-by-step implementation of the transformer model in PyTorch.
Check out [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar for a visual and intuitive explanation of how transformers work.
If you want to dive deeper into the mathematics and mechanics of transformers, consider reading the [Transformer paper](https://arxiv.org/abs/1706.03762) itself.
****************

Fave paper so far:

## [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004)

* This paper presents a compelling case that purported emergent abilities in LLMs are highly dependent on the metrics employed, challenging the community to reassess the foundational understanding of how LLMs evolve with scale.

****************
First favourites in research*****************

1. [Exploiting Novel GPT-4 APIs](https://arxiv.org/abs/2312.14302)

2. [Orca: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/pdf/2306.02707)

* _Can large language models (LLMs) train themselves?_
Credits: [Cameron Wolfe](https://cameronrwolfe.substack.com/p/orca-properly-imitating-proprietary) found through this [twitter thread](https://twitter.com/cwolferesearch/status/1673398297304911872)

****************

### Research focus

1) Alignment

-[A closer look at RLHF](https://twitter.com/cwolferesearch/status/1724486576992886985)

2)"Hallucination" problem

  -[Lately I have been thinking about the intersection between imagination and "hallucinations" in humans and machines respectively](https://ashioyajotham.substack.com/p/hallucinations-in-large-language)

3)Interpretability

    -[Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496)

4)Scaling Laws
    -[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.083613)

5)In-context learning

   -[In-Context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)

6)World Models

   -[Learning to Simulate Dynamic Environments with GameGAN](https://arxiv.org/abs/2005.12126)

   -[World Models](https://worldmodels.github.io/) by David Ha and Jürgen Schmidhuber.

    -[Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)

    -[V-JEPA: Latent Video Prediction for Visual Representation Learning](https://openreview.net/forum?id=WFYbBOEOtv)
