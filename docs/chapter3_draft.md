# Chapter 3: Giving LLMs Hands and Feet - Agents and Multi-Agent Systems

In the previous chapters, we explored the inner workings of Large Language Models and witnessed their impressive capabilities in tasks like text generation, translation, and even coding assistance. However, we also encountered their fundamental limitations: they are primarily text-in, text-out systems, inherently stateless, and lack the ability to interact directly with the outside world or execute complex, multi-step plans autonomously. If our personal assistant is only capable of answering questions based on its training data or the immediate conversation context, it won't be very helpful for tasks like booking appointments, searching the latest news, or controlling smart home devices.

This is where **AI Agents** enter the picture. Agents represent a significant leap forward, transforming LLMs from passive predictors into active participants capable of reasoning, planning, and interacting with their environment using tools. This chapter delves into the concept of agents, showing how we can empower LLMs to overcome their limitations and tackle more sophisticated problems. We'll build our first simple agent in Python and then explore how multiple agents can collaborate in **Multi-Agent Systems** to solve even more complex challenges, bringing our dream of a truly capable personal assistant closer to reality.

## Why Agents? Overcoming LLM Limitations

Basic LLM interactions often fall short for real-world applications:

1.  **Statelessness & Limited Context:** As discussed, LLMs forget everything once the context window is full or the session ends. They can't inherently remember user preferences across conversations or track progress on long tasks. [Source: AI Agents: Key Concepts and Overcoming Limitations of Large Language Models - LinkedIn]
2.  **Lack of Action/Grounding:** LLMs can *talk* about searching the web or calculating a number, but they can't *do* it themselves. Their knowledge is frozen at the time of training and they have no access to real-time information or external tools.
3.  **Difficulty with Complex Planning:** While LLMs can generate plans, executing them reliably, handling errors, and adapting to changing circumstances based on intermediate results is challenging for a single, monolithic prompt.

Agents are designed specifically to address these issues. They use an LLM as their core "brain" or reasoning engine but augment it with crucial components:

*   **Tools:** Access to external resources like search engines, databases, APIs, calculators, or even code execution environments.
*   **Memory:** Mechanisms to store and retrieve information beyond the LLM's limited context window, enabling state tracking and recall of past interactions.
*   **Planning & Reasoning Loop:** An execution framework that allows the agent to break down tasks, decide which tool to use, execute it, observe the result, and plan the next step iteratively.

## What is an AI Agent?

Think of an AI agent as an autonomous entity that perceives its environment (through user input, tool outputs, etc.), reasons about its goals, and acts upon that environment to achieve those goals. In the context of LLMs, an agent typically consists of:

1.  **LLM Core:** The language model acts as the reasoning engine, interpreting input, making decisions, and formulating plans or responses.
2.  **Tools:** A defined set of functions or APIs the agent *can* call. Examples include a web search tool, a calculator, a database query function, or a Python REPL (Read-Eval-Print Loop).
3.  **Agent Runtime/Executor:** A framework that orchestrates the interaction between the LLM, the tools, and the user input. It parses the LLM's desired action, calls the appropriate tool, feeds the result back to the LLM, and repeats until the task is complete.
4.  **Memory (Optional but common):** A component for storing conversation history, user preferences, or intermediate results, allowing the agent to maintain context over time. [Source: AI Agents: Key Concepts and Overcoming Limitations of Large Language Models - LinkedIn]

## The Agent Loop: Observe, Think, Act

Many agent frameworks implement a variation of the **ReAct (Reasoning and Acting)** approach. The agent operates in a loop:

1.  **Observe:** Receives user input or the result of a previous action.
2.  **Think (Reason):** The LLM analyzes the observation and the overall goal. It decides whether it has enough information to respond directly or if it needs to use a tool. If a tool is needed, it determines *which* tool and *what input* to provide to it.
3.  **Act:** If a tool is chosen, the agent executor calls that tool with the specified input. If no tool is needed, the LLM generates the final response.
4.  **(Repeat):** The output of the tool (the new observation) is fed back into the loop, starting the cycle again.

This iterative process allows the agent to break down complex tasks, gather information dynamically, and react to the results of its actions.

## Building a Simple Agent in Python (LangChain & LangGraph)

Let's make this concrete. We'll use **LangChain**, a popular open-source framework designed to simplify the development of LLM applications, and specifically its **LangGraph** extension for building robust, stateful agents. We'll create a simple agent that can chat and use a web search tool (Tavily Search).

**Components:**

*   **LLM:** We'll use a model capable of tool calling (e.g., Anthropic's Claude 3 Sonnet or OpenAI's GPT-4). Tool calling means the LLM is specifically trained to format its output in a way that indicates which tool it wants to use and with what arguments.
*   **Tools:** `TavilySearchResults` for web search.
*   **Memory:** `MemorySaver` from LangGraph to store conversation history.
*   **Agent Executor:** `create_react_agent` from LangGraph, which implements the ReAct logic using the LLM's tool-calling ability.

**Code Example:**

```python
# File: simple_agent.py
import os
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# --- 1. Setup: Environment Variables --- 
# Ensure your API keys are set as environment variables
# For Cursor/VS Code, you might use a .env file and python-dotenv
# or set them in your system environment
# Example: export ANTHROPIC_API_KEY="your_key_here"
# Example: export TAVILY_API_KEY="your_key_here"

# Check if keys are set (optional but good practice)
if "ANTHROPIC_API_KEY" not in os.environ or "TAVILY_API_KEY" not in os.environ:
    print("Error: API keys for Anthropic and Tavily must be set as environment variables.")
    exit()

# --- 2. Initialize Components --- 

# Memory: Stores conversation history per thread_id
memory = MemorySaver()

# LLM: Using Claude 3 Sonnet (requires langchain-anthropic)
# Ensure you have run: pip install langchain-anthropic
model = ChatAnthropic(model_name="claude-3-sonnet-20240229")

# Tools: Define the tools the agent can use
# Ensure you have run: pip install langchain-community tavily-python
search_tool = TavilySearchResults(max_results=3) # Get top 3 results
tools = [search_tool]

# --- 3. Create the Agent Executor --- 
# Uses LangGraph's prebuilt ReAct agent implementation
# Ensure you have run: pip install langgraph
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# --- 4. Interact with the Agent --- 

def run_agent_conversation():
    # Unique ID for this conversation thread
    # Allows memory to be specific to this chat
    conversation_id = "my_assistant_convo_01"
    config = {"configurable": {"thread_id": conversation_id}}

    print("Agent ready! Type 'quit' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        print("Agent: ...thinking...")
        # Stream the agent's response and intermediate steps
        # stream_mode="values" gives the full state at each step
        final_response = None
        try:
            for step in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config,
                stream_mode="values",
            ):
                # Print the type of message (Human, AI, Tool)
                # and its content for observing the agent's process
                last_message = step["messages"][-1]
                last_message.pretty_print() # Nicely formats the output
                print("-----") 
                # The final response is typically the last AI message
                if last_message.type == "ai" and not hasattr(last_message, 'tool_calls'):
                    final_response = last_message.content
            
            # Print the final answer clearly if available
            # Note: The loop above already prints intermediate steps
            # if final_response:
            #     print(f"\nAgent (Final): {final_response}")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_agent_conversation()

```

**Explanation:**

1.  **Setup:** We import necessary modules and ensure API keys are available as environment variables.
2.  **Components:** We initialize the `MemorySaver`, the `ChatAnthropic` LLM, and the `TavilySearchResults` tool.
3.  **Agent Creation:** `create_react_agent` wires everything together. It takes the LLM (which understands tool calling) and the list of available tools. Crucially, we link the `memory` using the `checkpointer` argument.
4.  **Interaction:**
    *   We define a `conversation_id` to keep track of this specific chat history.
    *   In a loop, we take user input.
    *   We `stream` the execution. This is powerful because it shows intermediate steps: the LLM's thoughts (often implicit in the tool call), the tool call itself, the tool's output, and the final AI response.
    *   `pretty_print()` helps visualize the different message types (Human, AI, Tool).
    *   The `config` dictionary passes the `thread_id` so `MemorySaver` knows which conversation history to load and save.

*   ***Vibe Coding Tips (Cursor):***
    *   **Setup:** Create `simple_agent.py`. Paste the code.
    *   **Installation:** If libraries are missing, Cursor might underline them. Use the chat (`@Cursor`) or terminal: `pip install langchain-anthropic langchain-community tavily-python langgraph python-dotenv`. Consider creating a `requirements.txt`.
    *   **API Keys:** Create a `.env` file in the same directory: 
        ```.env
        ANTHROPIC_API_KEY="sk-ant-...
        TAVILY_API_KEY="tvly-...
        ```
        Add `from dotenv import load_dotenv` and `load_dotenv()` at the top of your Python script.
    *   **Understanding:** Select `create_react_agent` and ask `@Cursor Explain what this function does`. Select the `stream` call and ask `@Cursor What are the different `stream_mode` options?`
    *   **Running:** Open Cursor's integrated terminal (`Ctrl+`` or `Cmd+```) and run `python simple_agent.py`.
    *   **Debugging:** If you get errors, select the traceback in the terminal or the problematic code line and ask `@Cursor Fix this error` or `@Cursor Why am I getting this error?`
    *   **Experimentation:** Try asking the agent questions that require search ("What's the latest news about AI?") and follow-up questions ("Who wrote the main article mentioned?") to see the memory and tool use in action.

This simple agent, combining reasoning (LLM), action (search tool), and memory, is far more capable than a basic LLM call and forms a building block for our personal assistant.

## Beyond One: Multi-Agent Systems

What happens when a task is too complex even for a single agent with tools? Maybe it requires coordinating multiple specialized skills, like researching a topic, writing code based on the research, and then generating a report.

This is where **Multi-Agent Systems** shine. Instead of one agent trying to juggle everything, we create multiple specialized agents that collaborate.

**Why Use Multi-Agent Systems?** [Source: Multi-agent Systems - LangGraph Docs]

*   **Modularity & Specialization:** Create expert agents (e.g., a `ResearcherAgent`, a `CodeWriterAgent`, a `ReportAgent`). Each is simpler to build, test, and maintain.
*   **Reduced Complexity:** A single LLM might get confused trying to manage too many tools or a very long, complex plan.
*   **Controlled Workflow:** Explicitly define how agents communicate and hand off tasks, allowing for more predictable and robust execution.

**Transforming Problems: Multi-Agent Architectures**

LangGraph allows us to define these systems as graphs where nodes are agents (or simple functions) and edges represent the flow of control and information.

*   **Supervisor:** A common pattern where a central 

"Supervisor" agent routes tasks to specialized agents (nodes) based on the current state or user request. Agents report back to the supervisor.
*   **Hierarchical:** Supervisors managing other supervisors for complex, layered tasks.
*   **Network:** Agents can potentially call any other agent (more flexible but can be harder to manage).

**Communication & State:**

*   **Shared State:** Agents communicate implicitly by reading from and writing to a shared state object (like `MessagesState` in LangGraph, which holds a list of messages).
*   **Handoffs:** Explicitly passing control and potentially specific data from one agent to another. LangGraph uses `Command(goto="agent_name", update={...})` for this, often triggered by an agent deciding its task is done and another agent needs to take over. [Source: Multi-agent Systems - LangGraph Docs]

**Practical Example: Multi-Agent Collaboration (Conceptual - LangGraph)**

Let's sketch out the structure for a system with a `ResearcherAgent` and a `CodeWriterAgent`, orchestrated by a `SupervisorAgent`.

```python
# File: multi_agent_system.py (Conceptual Structure)

# --- Imports (Similar to simple_agent.py + StateGraph) ---
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver # If needed
# ... other imports (LLM, Tools, Agent creation logic) ...

# --- Define State --- 
# More complex state might be needed than just messages
class AgentState(TypedDict):
    messages: list[BaseMessage]
    current_task: str
    research_notes: str | None
    code_snippet: str | None
    # ... other relevant state fields

# --- Agent Nodes (Functions representing agent logic) ---

def supervisor_node(state: AgentState) -> Command[Literal["researcher", "coder", END]]:
    # LLM decides which agent to route to next based on state["messages"] and state["current_task"]
    # Example Logic:
    # if research needed: return Command(goto="researcher", update=...)
    # if coding needed: return Command(goto="coder", update=...)
    # if task complete: return Command(goto=END, update=...)
    pass # Replace with actual supervisor logic

def researcher_node(state: AgentState) -> Command[Literal["supervisor"]]:
    # Researcher agent uses search tool based on state["current_task"]
    # research_agent = create_agent(...) # Using search tool
    # result = research_agent.invoke({"messages": state["messages"]})
    # notes = extract_notes_from_result(result)
    # Update state with notes and return control to supervisor
    # return Command(goto="supervisor", update={"research_notes": notes, "messages": result["messages"]})
    pass # Replace with actual researcher logic

def coder_node(state: AgentState) -> Command[Literal["supervisor"]]:
    # Coder agent uses Python REPL or code generation prompt 
    # based on state["current_task"] and state["research_notes"]
    # coder_agent = create_agent(...) # Using python_repl
(Content truncated due to size limit. Use line ranges to read in chunks)