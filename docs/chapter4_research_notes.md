# Chapter 4 Research Notes: Fine-Tuning LLMs

## Source: Fine-tuning large language models (LLMs) in 2025 | SuperAnnotate (https://www.superannotate.com/blog/llm-fine-tuning)

**What is Fine-Tuning?**

*   **Definition:** The process of taking pre-trained models and further training them on smaller, specific datasets to refine their capabilities and improve performance in a particular task or domain.
*   **Purpose:** To bridge the gap between generic pre-trained models and the unique requirements of specific applications, ensuring the model aligns closely with desired outcomes (e.g., adapting GPT-3 for medical report generation).
*   **Mechanism:** Unlike pre-training (unsupervised on vast unstructured data), fine-tuning is typically a supervised learning process using labeled examples (often prompt-response pairs) to update the model's weights.
*   **Process:**
    1.  Prepare a specific, labeled dataset (instruction dataset).
    2.  Divide data into training, validation, and test splits.
    3.  Pass prompts from the training data to the LLM.
    4.  Calculate the error/difference between the model's predictions and the actual labels.
    5.  Use the error to adjust model weights (typically via gradient descent), making weights more responsible for errors adjusted more, and less responsible ones adjusted less.
    6.  Repeat over multiple iterations (epochs) to minimize error and adapt the model to the nuances of the new dataset.

**When to Use Fine-Tuning (vs. Prompt Engineering/In-Context Learning):**

*   **In-Context Learning (ICL):** Improving performance via specific task examples *within the prompt* (zero-shot, one-shot, few-shot). Doesn't update model weights. Useful but can fail (especially for smaller LLMs) and consumes valuable context window space.
*   **Fine-Tuning:** Use when ICL isn't sufficient. Fine-tuning *updates model weights* based on a dataset of labeled examples, leading to better completion of specific tasks.

**Methods for Fine-Tuning:**

1.  **Supervised Fine-Tuning (SFT):** The general process described above, using labeled data (prompt-completion pairs) for a specific task.
2.  **Instruction Fine-Tuning:** A type of SFT where the dataset consists of examples demonstrating how the model should respond to specific instructions (e.g., "Summarize the following text:", "Translate this text:"). Helps the model learn to follow instructions better.
3.  **Full Fine-Tuning:** Updating *all* of the model's weights during the fine-tuning process. Requires significant memory and compute resources (similar to pre-training) to store model weights, gradients, optimizer states, etc. Can lead to "catastrophic forgetting" where the model performs well on the new task but degrades on others. Results in a new, full-sized model version for each task.
4.  **Parameter-Efficient Fine-Tuning (PEFT):**
    *   **Goal:** Reduce computational cost and memory requirements of fine-tuning.
    *   **Mechanism:** Updates only a *small subset* of the model's parameters, freezing the rest. Techniques like LoRA can reduce trainable parameters drastically (e.g., by 10,000x).
    *   **Benefits:** More manageable memory requirements, mitigates catastrophic forgetting (as original weights are mostly untouched), avoids creating large new model versions for each task (smaller adapter layers are saved).
5.  **Transfer Learning:** A broader concept where knowledge from a model trained on a large, general dataset is transferred to a new, related task with a smaller, specific dataset. Fine-tuning is a common way to achieve transfer learning for LLMs.
6.  **Task-Specific Fine-Tuning:** Fine-tuning on a dataset designed for a single, specific task (e.g., sentiment analysis, translation). Can achieve high performance on that task but risks catastrophic forgetting.
7.  **Multi-Task Fine-Tuning:** Fine-tuning on a mixed dataset containing examples for *multiple* tasks simultaneously. Aims to improve performance across all included tasks and avoid catastrophic forgetting. May require large datasets (e.g., 50k-100k examples).
8.  **Sequential Fine-Tuning:** Adapting a model sequentially across related tasks, becoming progressively more specific (e.g., General -> Medical -> Pediatric Cardiology).
9.  **Other Types:** Adaptive, behavioral, reinforced fine-tuning (e.g., RLHF - Reinforcement Learning from Human Feedback).

**Fine-Tuning vs. Retrieval-Augmented Generation (RAG):**

*   **RAG:** Grounds the model using external, up-to-date knowledge sources/documents retrieved based on the query. Provides sources, bridges the gap between general knowledge and specific, current information. Good for situations where facts evolve. Allows easy updates/removal of knowledge without retraining.
*   **Fine-Tuning:** Embeds knowledge *into the model's architecture* by updating weights. Better for teaching the model specific styles, formats, or tasks where the underlying reasoning needs adjustment, not just factual recall.
*   **Combined Use:** RAG and fine-tuning are not mutually exclusive. Fine-tuning can be applied *to RAG systems* (e.g., fine-tuning the retriever or the generator component) to improve weaker parts and enhance overall performance.


