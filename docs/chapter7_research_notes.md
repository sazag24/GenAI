# Chapter 7 Research Notes: Memory Management

## Source: Memory, Context, and Cognition in LLMs (https://promptengineering.org/memory-context-and-cognition-in-llms/)

*   **Misconception:** LLMs are often thought to have infinite memory, like an all-knowing oracle.
*   **Reality:** They operate more like a forgetful genius.
*   **Context Window:** This is the LLM's short-term memory (like RAM). It's the amount of information the model can hold and process at any given moment.
*   **Permanent Memory (Analogy):** The vast training data can be thought of as permanent memory, but the LLM can only access a small portion of it through the context window at a time.
*   **Functioning:** The LLM uses the context window to identify keywords, map them to its knowledge base, and activate relevant entities/relationships to generate responses.
*   **Limitation:** The finite size of the context window means the LLM cannot hold all necessary primary and secondary information simultaneously for complex tasks, akin to seeing only part of an equation.
*   **Importance of Context:** Providing sufficient context, keywords, and examples in prompts is crucial for relevant outputs.
*   **Workarounds:** Techniques like Chain of Thought (CoT) help guide the LLM step-by-step, breaking down problems and allowing it to effectively use its context window to simulate deeper reasoning or access different 'bits' of information sequentially.
*   **Analogy:** Like having many reference books but only being able to open one at a time. CoT helps the LLM 'consult' different 'books' (information pieces) sequentially.
*   **Key Distinction:** Context window is the *immediate processing capacity* (short-term/working memory), distinct from the broader concept of memory which might involve external storage or retrieval mechanisms (discussed later).




## Source: Conversation Summary Buffer | 🦜️🔗 LangChain (https://python.langchain.com/v0.1/docs/modules/memory/types/summary_buffer/)

*   **Concept:** Combines a buffer of recent interactions with a summary of older interactions.
*   **Mechanism:** Keeps recent messages directly in memory up to a `max_token_limit`.
*   **Summarization:** When the buffer exceeds the token limit, older messages are compiled into a summary, which is then included along with the remaining recent messages.
*   **Benefit:** Provides a balance between retaining recent detail and keeping a condensed history of the entire conversation, preventing the context from growing indefinitely while still maintaining long-term context.
*   **Usage:** Can be used directly or within a LangChain `ConversationChain`.
*   **Output:** Can return history as a formatted string or a list of messages (useful for chat models).




## Source: Augmenting LLMs with Retrieval, Tools, and Long-term Memory (https://medium.com/infinitgraph/augmenting-llms-with-retrieval-tools-and-long-term-memory-b9e1e6b2fc28)

### Key Concepts:

*   **Long-Term Memory:** Addresses LLM limitations in retaining information beyond the context window.
*   **Vector Databases:** Store embeddings (numerical representations) of data chunks. Used for semantic search to retrieve relevant information.
*   **Knowledge Graphs:** Represent information as entities and relationships, offering structured knowledge retrieval.
*   **Data Preprocessing for Vector DBs:** Involves cleaning, standardizing, and "chunking" data into smaller pieces before embedding.
*   **Retrieval Process:** User query is embedded, compared against embeddings in the vector DB using similarity measures (e.g., cosine similarity), and relevant chunks are retrieved.
*   **Augmentation:** Retrieved information is added to the LLM's context/prompt along with the original query.
*   **Tools:** LLMs can be taught to use external tools (APIs, calculators, search engines) to access real-time information or perform specific tasks (e.g., Toolformer).
*   **Example:** The article provides a hands-on example using FAISS (vector store) and Google's Generative AI models to build a RAG system.

### Comparison:

*   **Vector Stores:** Good for semantic similarity search over large text corpora.
*   **Knowledge Graphs:** Better for structured data and querying specific relationships between entities.

### Implementation Notes (from article example):

*   Uses FAISS for indexing and retrieval.
*   Uses Google's `text-embedding-004` for embeddings.
*   Uses `gemini-1.5-flash` as the LLM.
*   Demonstrates steps: install libraries, load/preprocess documents (chunking), embed documents, create FAISS index, implement retrieval function, implement RAG function.


