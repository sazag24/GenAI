# Chapter 1 Research Notes: LLM Fundamentals

## Source: An Intuitive Explanation of Large Language Models (https://amistrongeryet.substack.com/p/large-language-models-explained)

**Core Concept:**
*   LLMs (like GPT) are neural networks trained to predict the next word in a sequence.
*   Input: A sequence of words.
*   Output: Prediction of the most likely next word.

**Training:**
*   Requires enormous amounts of training data (e.g., Wikipedia, books, web text).
*   The "predict next word" task allows leveraging vast, easily obtainable text data.
*   Training process involves adjusting connection weights in the neural network based on examples.
*   The network learns patterns in language (grammar, style, facts) to improve predictions.
*   Example: Learning subject-verb agreement (singular noun -> verb ending in "s").
*   The process involves identifying useful circuits and tuning them; sometimes described as "alchemy" due to lack of full understanding.

**How They Work (Simplified Neural Net View):**
*   Nodes (neurons) have activation levels.
*   Connections between nodes have weights.
*   Activation level of a node is calculated based on weighted sums of activations from connected nodes.
*   Training adjusts weights to make the network better at predicting the correct next word for a given input sequence.

**Capabilities & Emergent Behavior:**
*   Can imitate styles by activating learned patterns associated with that style (e.g., Shakespearean).
*   Can follow instructions (in plain English) because the training data includes examples of instruction following.
*   Instruction following is often strengthened via Reinforcement Learning from Human Feedback (RLHF).
*   Can perform tasks in "zero-shot" (no specific examples given for the task) or "few-shot" (a few examples provided in the prompt) settings.

**Limitations:**
*   **Statelessness/Lack of Deep Understanding:**
    *   Can "hallucinate" (make up facts).
    *   Can be inconsistent or easily confused.
    *   May not truly "understand" concepts, just statistical patterns.
    *   Conceptual errors might only affect a few words, making them hard for the word-by-word training process to correct effectively.
*   **Feedforward Nature:**
    *   Current LLMs are mostly feedforward networks (information flows one way).
    *   They lack loops (recurrence) like human brains, limiting their ability to "stop and think" or self-reflect during generation.
    *   Processing for each word occurs in a single pass through the network.
*   **Memory:**
    *   No inherent long-term memory beyond the training data.
    *   Limited short-term memory via the "token buffer" (context window), which has size limits (e.g., ~25,000 words for GPT-4 mentioned in article).
*   **Other Issues:**
    *   Difficulty distinguishing instructions from external information vs. owner instructions (potential for "jailbreaking").
    *   Potential for unexpected behavior like the "Waluigi effect" (model behaving opposite to its training).

**Why They Don't Always Get Stuck (Planning Ahead):**
*   Even without loops, a single pass through the deep network allows for some level of "planning" for the next word, considering broader context.
*   Output tends to be somewhat generic or bland, reducing chances of painting into a corner.
*   Ability to find *something* to say even if the context becomes awkward.





## Source: Transformer (deep learning architecture) - Wikipedia (https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture))

**Core Concept:**
*   Transformer architecture developed by Google researchers (2017 paper "Attention Is All You Need").
*   Based on multi-head attention mechanism.
*   Replaced recurrent units (like RNNs, LSTMs), allowing for parallel processing and faster training.

**History & Evolution:**
*   **RNNs (e.g., Elman network, 1990):** Processed sequences token by token, suffered from vanishing gradients for long sequences.
*   **LSTMs (1995/1997):** Overcame vanishing gradients using gates and multiplicative units (related to attention), became standard for long sequences but still processed sequentially.
*   **Seq2Seq (Encoder-Decoder) Models (c. 2014):** Used RNNs (LSTM/GRU) to map input sequence to a fixed-size vector (encoder) and then generate output sequence (decoder). Suffered from information bottleneck with long inputs.
*   **Attention in Seq2Seq (e.g., RNNsearch, 2014):** Allowed decoder to look back at relevant parts of the input sequence, solving the bottleneck problem.
*   **Parallelizing Attention (2016-2017):** Self-attention mechanisms applied to feedforward networks, removing the need for recurrence and enabling parallelization (key idea for Transformer).
*   **Transformer (2017):** Introduced the multi-head self-attention mechanism, fully parallelizable, became the foundation for modern LLMs.
*   **Post-Transformer Era (AI Boom):**
    *   **ELMo (2018):** Contextualized word embeddings.
    *   **BERT (2018):** Transformer encoder-only model, pre-trained on masked language modeling and next-sentence prediction.
    *   **GPT Series (2018-present):** Transformer decoder-only models, pre-trained on autoregressive language modeling (predicting the next token). Led to models like ChatGPT.
    *   Transformers applied beyond NLP: Vision (ViT), audio, multimodal (DALL-E, Sora), robotics, etc.

**Architecture Components:**
*   **Tokenization:** Converts text into numerical tokens (e.g., using Byte Pair Encoding, WordPiece).
*   **Embedding:** Converts each token into a vector representation (lookup table).
*   **Positional Encoding:** Adds information about the position of tokens in the sequence, as the architecture itself doesn't process sequentially. Uses sinusoidal functions or learned embeddings.
*   **Attention Mechanism (Scaled Dot-Product Attention):**
    *   Calculates attention scores between tokens based on Query (Q), Key (K), and Value (V) matrices derived from token embeddings.
    *   Score = softmax( (Q * K^T) / sqrt(d_k) ) * V
    *   Allows model to weigh the importance of different tokens when representing a specific token.
*   **Multi-Head Attention:** Runs the attention mechanism multiple times in parallel with different learned projections (different Q, K, V matrices). Allows the model to focus on different aspects of the sequence simultaneously. Outputs are concatenated and linearly projected.
*   **Feedforward Network:** Applied independently to each position after the attention layer. Typically a 2-layer MLP with ReLU or GeLU activation.
*   **Residual Connections:** Add the input of a sub-layer (Attention or FFN) to its output (X + Sublayer(X)). Helps with gradient flow and enables deeper networks.
*   **Layer Normalization:** Applied before (Pre-LN) or after (Post-LN) each sub-layer. Stabilizes training.
*   **Encoder:** Stack of identical layers, each with Multi-Head Self-Attention and Feedforward Network sub-layers. Processes the input sequence.
*   **Decoder:** Stack of identical layers, each with Masked Multi-Head Self-Attention, Multi-Head Cross-Attention (attending to encoder output), and Feedforward Network sub-layers. Generates the output sequence autoregressively.
*   **Masked Self-Attention (Decoder):** Prevents decoder from attending to future tokens during training/generation.

**Training:**
*   Typically uses self-supervised learning on large datasets (e.g., The Pile).
*   Common pre-training tasks:
    *   **Masked Language Modeling (MLM):** Used by BERT. Mask out some input tokens and predict them.
    *   **Autoregressive Language Modeling:** Used by GPT. Predict the next token given previous tokens.
*   Followed by **Fine-tuning** on smaller, task-specific datasets (e.g., question answering, sentiment analysis).
*   Training requires stabilization methods (e.g., learning rate warmup, layer normalization).


