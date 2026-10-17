"""Generate the small offline demo corpus (illustrative abstracts, written for this project,
NOT real publications) and labelled evaluation queries. Use build_corpus.py for real papers."""
import json
from parse.corpus import Paper, save_corpus

T = {
 "few_shot": [
  ("Learning New Categories from Handfuls of Labelled Images", "We study how a classifier can acquire previously unseen categories when only a handful of labelled images per class are available. A meta-trained prototype network adapts rapidly and outperforms fine-tuning baselines."),
  ("Episodic Meta-Training for Low-Data Recognition", "Models are trained across many small tasks so that adaptation to a novel task needs only a few gradient steps. Experiments show strong generalisation when supervision is extremely scarce."),
  ("Metric Spaces for Data-Scarce Classification", "We learn an embedding where distance to a support set of a few examples determines class membership. The method reaches high accuracy with one to five samples per category."),
  ("Transferring Knowledge to Rare Classes", "Rare categories with very few training instances are hard for deep networks. We propose feature hallucination that augments minority classes and improves recognition."),
 ],
 "gnn": [
  ("Message Passing on Relational Structures", "We propose a neural architecture that aggregates information from neighbouring nodes in a network to produce node representations. It achieves strong results on citation and social network benchmarks."),
  ("Spectral Convolutions over Irregular Domains", "Convolution is generalised to data defined on nodes and edges using the eigenbasis of the graph Laplacian. The approach classifies vertices with limited labels."),
  ("Attention Mechanisms for Networked Data", "A learned attention weight is assigned to each neighbour of a vertex so that more informative connections dominate aggregation. Accuracy improves on molecular and citation graphs."),
  ("Predicting Links in Large Interaction Networks", "We embed entities and their relations to forecast missing edges in knowledge bases and social platforms using stacked neighbourhood aggregation."),
 ],
 "protein": [
  ("Predicting Three-Dimensional Folds from Amino Acid Sequences", "A deep network infers the spatial arrangement of residues directly from the sequence of a polypeptide chain. Predicted structures approach experimental accuracy."),
  ("Co-evolution Signals for Residue Contact Estimation", "Correlated mutations across homologous families reveal which residues are close in space. We use these signals to estimate tertiary structure of biomolecules."),
  ("Neural Models of Biomolecular Geometry", "We design an equivariant architecture that reasons about atomic coordinates and outputs the native conformation of macromolecules, accelerating drug discovery."),
  ("Language Models Trained on Evolutionary Sequence Data", "Self-supervised pretraining on millions of protein sequences yields representations that capture structure and function without alignments."),
 ],
 "mt": [
  ("Attention-Based Sequence Models for Cross-Lingual Text Conversion", "We present an encoder-decoder system that converts sentences from one human language into another using learned alignment. It yields higher BLEU than phrase-based systems."),
  ("Translating Between Languages Without Parallel Corpora", "Unsupervised methods map monolingual embeddings of two languages into a shared space, enabling sentence conversion with no aligned examples."),
  ("Subword Units for Open-Vocabulary Language Conversion", "Rare and unseen words are handled by splitting them into smaller units, improving the rendering of morphologically rich languages."),
  ("Multilingual Pretraining for Low-Resource Language Pairs", "A single model trained on many languages transfers to pairs with little bilingual text, raising output quality for under-resourced languages."),
 ],
 "seg": [
  ("Pixel-Wise Labelling of Street Scenes", "A fully convolutional architecture assigns a semantic class to every pixel in an urban image, delineating roads, vehicles and pedestrians."),
  ("Delineating Organs in Volumetric Medical Scans", "An encoder-decoder with skip connections outlines anatomical structures in tomography volumes, supporting radiotherapy planning with limited annotated data."),
  ("Instance Masks with Region Proposals", "We extend a detection framework to predict a binary mask for each object, separating overlapping items in cluttered photographs."),
  ("Boundary-Aware Partitioning of Satellite Imagery", "Land-cover regions are partitioned into coherent segments using boundary refinement, improving mapping of fields and buildings from aerial views."),
 ],
 "rl": [
  ("Learning Control Policies through Trial and Error", "An agent improves its decision rule by interacting with an environment and maximising cumulative reward. We demonstrate mastery of arcade games from raw pixels."),
  ("Value Estimation with Deep Function Approximators", "A neural network approximates expected return for each action, stabilised by replay of past experience and slowly updated target networks."),
  ("Policy Gradient Methods with Trust Regions", "We constrain each update of a stochastic policy to remain close to the previous one, achieving reliable improvement on locomotion tasks."),
  ("Exploration Bonuses for Sparse-Reward Environments", "Curiosity-driven intrinsic signals encourage an agent to visit novel states when external feedback is rare, solving hard-exploration problems."),
 ],
 "fl": [
  ("Training Models Across Decentralised Devices", "Many phones jointly train a shared model while raw user data stays on each device; only parameter updates are averaged by a coordinating server."),
  ("Protecting Individuals in Collaborative Learning", "We add calibrated noise to gradients so that no single person's record can be inferred from a released model, giving formal privacy guarantees."),
  ("Communication-Efficient Distributed Optimisation", "Compressing and sparsifying updates reduces bandwidth when thousands of clients with heterogeneous data take part in training."),
  ("Secure Aggregation without Revealing Individual Contributions", "Cryptographic masking lets a server compute the sum of client updates without seeing any single one, safeguarding confidential information."),
 ],
 "asr": [
  ("End-to-End Transcription of Spoken Utterances", "A single neural network maps acoustic waveforms directly to text without a hand-built pronunciation lexicon, lowering word error rate on conversational audio."),
  ("Self-Supervised Representations of Raw Audio", "Pretraining on unlabelled recordings yields features that make recognising spoken words possible with only a few hours of transcribed data."),
  ("Robust Recognition of Speech in Noisy Environments", "Data augmentation and enhancement front-ends improve transcription accuracy when background noise and reverberation corrupt the signal."),
  ("Streaming Decoding for Voice Assistants", "A low-latency model converts speech to text incrementally as the user talks, enabling responsive voice interfaces on mobile hardware."),
 ],
}
papers = []
for topic, items in T.items():
    for i, (title, abs_) in enumerate(items):
        papers.append(Paper(id=f"sample:{topic}-{i}", title=title, abstract=abs_,
                            authors=["Demo Author"], year=2018 + i, url="", source="sample", topic=topic))
save_corpus(papers, "data/sample_papers.json")

Q = [
 ("how can a model recognise objects it has barely seen before", ["few_shot"]),
 ("systems that reason over nodes connected by edges", ["gnn"]),
 ("working out the shape of biological molecules from their building blocks", ["protein"]),
 ("automatically rendering text from one tongue into another", ["mt"]),
 ("marking every region of a photograph with what it depicts", ["seg"]),
 ("software that gets better at a game by being rewarded for good moves", ["rl"]),
 ("learning together without sharing personal data", ["fl"]),
 ("turning what people say aloud into written words", ["asr"]),
 ("neural networks on graphs", ["gnn"]),
 ("speech recognition", ["asr"]),
]
json.dump([{"query": q, "relevant_topics": r} for q, r in Q], open("data/eval_queries.json", "w"), indent=2)
print(len(papers), "papers,", len(Q), "queries")
