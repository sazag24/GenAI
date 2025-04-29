# Chapter 6 Research Notes: Context Management

## Source: Mastering State in Stateless LLMs - Luminis (https://www.luminis.eu/blog/mastering-state-in-stateless-llm/)

**LLM Statelessness:**

*   Large Language Models (LLMs) operate without inherent memory; they are **stateless** by design.
*   Each interaction is independent from the previous ones unless the history is explicitly provided.

**Context Window:**

*   The only "state" an LLM recognizes during an interaction is the **context** provided in the current input.
*   This context typically includes the user's input messages and the model's previous responses within that session.
*   The **context window** refers to the maximum amount of text (measured in tokens) that the model can process at once.
*   The LLM parses the entire context provided within the window to generate its response.
*   A larger context window allows the model to consider more of the conversation history or provided background information.

**Managing State (Example Approach):**

*   Since LLMs are stateless, managing state for multi-turn conversations or tasks requires external mechanisms.
*   One approach involves using a **structured format (like JSON)** to represent the required state.
*   This state object can be updated based on user input and LLM output in separate steps:
    1.  **Extract State Change:** An LLM call analyzes the user input and the current state (provided as JSON) to determine necessary changes (e.g., set a field, append to a list).
    2.  **Apply Change:** The application updates the state object based on the extracted changes.
    3.  **Generate Next Response/Question:** Another LLM call uses the *updated* state (provided as JSON) to determine the next appropriate response or question, aiming to gather missing information or proceed with the task.
*   Frameworks/libraries (like OpenAI's Python client with Pydantic or Ollama's structured output) can simplify working with structured state objects (defining schemas, parsing responses).
*   This structured approach provides more control and predictability compared to simply instructing the LLM to remember things in natural language.


