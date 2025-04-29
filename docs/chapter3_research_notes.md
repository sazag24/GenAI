# Chapter 3 Research Notes: Agents and Multi-Agent Systems

## Source: AI Agents: Key Concepts and Overcoming Limitations of Large Language Models - LinkedIn (https://www.linkedin.com/pulse/ai-agents-key-concepts-overcoming-limitations-large-language-dubov-lvogc)

**Motivation for Agents (Overcoming LLM Limitations):**

*   **Lack of Contextual Awareness:** LLMs struggle to maintain context over long interactions (stateless nature). Agents address this using persistent memory and state tracking, allowing them to remember context and handle complex, multi-step tasks.
*   **Lack of Task Specialization:** LLMs are often generalists, lacking deep expertise in specific domains. Agents can be specialized and fine-tuned for particular tasks or industries (e.g., healthcare, finance), providing more accurate and relevant outputs.
*   **Lack of Autonomous Decision-Making:** LLMs primarily generate text based on prompts and lack true autonomy. Agents are designed for autonomous decision-making using techniques like reinforcement learning. They can operate independently based on goals and real-time data, reducing the need for constant human supervision.

**What are AI Agents?**

*   Autonomous entities designed to perform tasks and make decisions based on their environment and goals.
*   Go beyond LLMs (used for NLP) by incorporating contextual understanding, persistent memory, state tracking, and decision-making capabilities.
*   Can adapt, learn, and evolve autonomously, unlike traditional AI systems relying on predefined rules.

**Agent Capabilities & Applications:**

*   **Synthetic User Research:** Simulate user interactions to gather insights without traditional methods.
*   **Customer Service:** Handle inquiries autonomously, provide personalized responses, remember past interactions.
*   **Healthcare:** Assist in diagnosis, treatment suggestions, patient data management.
*   **Organizational Knowledge Management:** Manage and utilize organizational knowledge effectively.

**Challenges:**

*   Ethical use.
*   Managing biases.
*   Protecting user privacy.
*   Integration into existing systems.





## Source: Build an Agent - LangChain Python Docs (https://python.langchain.com/docs/tutorials/agents/)

**Core Concept:**

*   Agents use an LLM as a reasoning engine to decide which actions (tools) to take based on user input.
*   The results of actions are fed back to the LLM to determine the next step or finish.
*   Often achieved via "tool-calling" capabilities of modern LLMs.

**Simple Agent Example (LangChain using `create_react_agent`):**

*   **Goal:** Build an agent that can use a search tool (Tavily) and has conversational memory.
*   **Components:**
    *   **LLM:** The reasoning engine (e.g., `ChatAnthropic` with Claude 3 Sonnet).
    *   **Tools:** Functions the agent can call (e.g., `TavilySearchResults`). A list of tools is provided to the agent.
    *   **Agent Executor:** The runtime that orchestrates the LLM, tools, and memory (using `create_react_agent` from `langgraph.prebuilt`). `create_react_agent` specifically implements the ReAct (Reasoning and Acting) framework.
    *   **Memory:** To maintain conversation history (e.g., `MemorySaver` from `langgraph.checkpoint.memory`). The `checkpointer` argument links memory to the agent executor.
    *   **Configuration (`config`):** Used to manage state, especially the `thread_id` for conversation memory.

*   **Basic Code Structure:**

```python
# 1. Import necessary libraries
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# 2. Set up memory
memory = MemorySaver()

# 3. Initialize the LLM
# Ensure API keys are set as environment variables (e.g., ANTHROPIC_API_KEY, TAVILY_API_KEY)
model = ChatAnthropic(model_name="claude-3-sonnet-20240229")

# 4. Define tools the agent can use
search = TavilySearchResults(max_results=2)
tools = [search]

# 5. Create the agent executor
# This uses the ReAct logic: LLM reasons about which tool to use, uses it, observes result, reasons again.
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# 6. Define a configuration for the conversation thread
config = {"configurable": {"thread_id": "user_conversation_123"}}

# 7. Interact with the agent (streaming example)
# First message (establishes context)
print("--- User: hi im bob! and i live in sf ---")
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

print("\n--- User: whats the weather where I live? ---")
# Second message (agent uses memory and tools)
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

```

*   **Explanation:**
    *   The agent receives the human message.
    *   The LLM (via `create_react_agent`) decides if a tool is needed. If yes, it outputs a `tool_use` message specifying the tool and arguments (e.g., `tavily_search_results_json` with query `san francisco weather`).
    *   The agent executor runs the tool and gets the results.
    *   The results are passed back to the LLM in a `ToolMessage`.
    *   The LLM generates the final response to the user based on the tool results and conversation history.
    *   Memory (`MemorySaver`) automatically stores the interaction history associated with the `thread_id`.

**Vibe Coding / Cursor Integration Note:**

*   When writing this code in Cursor, you would:
    1.  Create a new Python file (e.g., `simple_agent.py`).
    2.  Paste or type the import statements.
    3.  Ensure necessary libraries (`langchain-anthropic`, `langchain-community`, `langgraph`, `tavily-python`) are installed. You can ask Cursor to help install them (e.g., `@Cursor install langchain-anthropic langchain-community langgraph tavily-python`).
    4.  Set up environment variables for API keys (e.g., `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`). Cursor might prompt you or you can manage this via your terminal or IDE settings.
    5.  Write the rest of the code, potentially using Cursor's AI features (@Cursor) to explain parts, suggest alternatives, or debug.
    6.  Run the script directly from Cursor's integrated terminal.
    7.  Observe the streaming output in the terminal, showing the agent's reasoning steps (tool calls) and final responses.





## Source: Multi-agent Systems - LangGraph Docs (https://langchain-ai.github.io/langgraph/concepts/multi_agent/)

**Why Multi-Agent Systems?**

*   Break down complex problems that are hard for a single agent to manage.
*   Reasons to use:
    *   **Too many tools:** Single agent struggles to choose the right tool.
    *   **Complex context:** Context window becomes too large or complex for one agent.
    *   **Need for specialization:** Require multiple experts (e.g., planner, researcher, coder).
*   **Benefits:**
    *   **Modularity:** Easier development, testing, maintenance.
    *   **Specialization:** Create expert agents for better performance.
    *   **Control:** Explicitly define communication flows.

**Transforming Problems into Multi-Agent Setups (Architectures):**

*   **Core Idea:** Represent agents as nodes in a graph (using LangGraph).
*   **Architectures:**
    *   **Network:** Any agent can communicate with any other agent (many-to-many). Good for non-hierarchical problems.
    *   **Supervisor:** A central agent directs traffic, deciding which specialist agent to call next. Agents report back to the supervisor.
    *   **Supervisor (Tool-Calling):** A variation where the supervisor LLM uses tool-calling to invoke other agents (which are defined as tools).
    *   **Hierarchical:** A supervisor manages other supervisors, creating layers of control. Allows for more complex workflows.
    *   **Custom:** Define specific communication paths; only certain agents can call others.

**Multi-Agent Concepts (Collaboration, Roles, Communication):**

*   **Roles:** Implicit in specialization (e.g., planner agent, research agent, coding agent).
*   **Collaboration/Communication (Handoffs):** How agents pass control and information.
    *   Agents (nodes in the graph) decide whether to finish or route to another agent.
    *   **Handoffs:** Mechanism for transferring control.
        *   Specify **destination** (target agent/node).
        *   Specify **payload** (information/state update to pass).
    *   **Implementation in LangGraph:**
        *   Agent nodes can return a `Command` object.
        *   `Command(goto="agent_name", update={"state_key": "value"})` specifies the next agent and updates the shared state.
        *   For agents within subgraphs communicating with agents in a parent graph, use `graph=Command.PARENT`.
    *   **Handoffs as Tools:** Wrap the handoff logic (returning a `Command`) within a tool definition. This allows a ReAct-style agent to decide to hand off control to another agent via a tool call.
*   **Communication via State:** How agents share information.
    *   **Graph State:** The shared memory or state of the multi-agent system.
    *   **Strategies for Sharing:**
        *   **Different State Schemas:** Each agent might operate on a different part of the state.
        *   **Shared Message List:** All agents append to and read from a common list of messages (like `MessagesState`).
        *   **Share Full History:** Pass the entire state history between agents.
        *   **Share Final Result:** Only pass the final output of one agent to the next.





## Source: LangGraph Multi-Agent Collaboration Tutorial (https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/multi-agent-collaboration.ipynb)

**Practical Example: Multi-Agent Network (Researcher + Chart Generator)**

*   **Goal:** Create a system where a user asks a question, a researcher agent finds information using a search tool, and if needed, a chart generator agent creates a chart using a Python REPL tool.
*   **Architecture:** Network/Supervisor (Implicit). The flow is somewhat directed (User -> Researcher -> Chart Generator or END), but agents communicate and decide the next step.

*   **Components:**
    *   **LLM:** Reasoning engine (e.g., `ChatAnthropic`).
    *   **Tools:**
        *   `TavilySearchResults`: For the researcher agent.
        *   `PythonREPL`: For the chart generator agent (to execute code for plotting).
    *   **State:** `MessagesState` (a list of messages) to hold the conversation history and intermediate results shared between agents.
    *   **Agent Nodes:** Functions representing each agent's logic.
        *   Each node takes the current state (`MessagesState`) as input.
        *   Invokes its specific agent logic (often using `create_react_agent` or similar).
        *   Returns a `Command` object to update the state (e.g., add the agent's response to the `messages` list) and specify the next node (`goto`).
    *   **Graph:** `StateGraph` defines the structure.
        *   Nodes are added (`add_node`).
        *   Edges define the flow (`add_edge`, `add_conditional_edges`). `START` edge points to the initial agent (researcher).
        *   Conditional edges (`add_conditional_edges`) are used to route based on agent output (e.g., does the researcher need the chart generator, or is the task finished?).

*   **Key Code Snippets (Conceptual):**

```python
from typing import Annotated, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
import os

# --- Setup (API Keys, Tools, LLM) ---
os.environ["ANTHROPIC_API_KEY"] = "YOUR_API_KEY" # Replace with actual key or env var loading
os.environ["TAVILY_API_KEY"] = "YOUR_API_KEY"

tavily_tool = TavilySearchResults(max_results=5)
python_repl_tool = PythonREPL() # Warning: Executes code locally!

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# --- Agent Creation Utility ---
def create_agent(llm, tools, system_message: str):
    # Simplified version of using create_react_agent
    # In the tutorial, it uses a prompt template and binds tools
    # This function would encapsulate that logic
    return create_react_agent(llm, tools, messages_modifier=system_message)

# --- Agent Nodes ---

# Function to decide the next step (route)
def get_next_node(state: MessagesState) -> Literal["chart_generator", "__end__"]:
    last_message = state["messages"][-1]
    # Simple routing logic based on content (example)
    if "generate chart" in last_message.content.lower():
        return "chart_generator"
    else:
        return END # Use END constant from langgraph.graph

# Researcher Agent Node
def research_node(state: MessagesState) -> Command[Literal["chart_generator", END]]:
    research_agent = create_agent(
        llm,
        [tavily_tool],
        system_message="You are a research assistant. You can use the search tool."
    )
    result = research_agent.invoke(state)
    # Determine next step based on result
    goto = get_next_node(result) # Logic to decide if chart is needed
    # Wrap the response in a HumanMessage for the next agent if needed
    # Important: The tutorial shows adding the agent's response to the state's message list
    return Command(
        goto=goto,
        update={"messages": result["messages"]} # Append agent's output to state
    )

# Chart Generator Agent Node
def chart_node(state: MessagesState) -> Command[Literal["researcher", END]]: # Can potentially loop back
    chart_agent = create_agent(
        llm,
        [python_repl_tool],
        system_message="You are a chart generator. Use the python_repl tool to generate charts."
    )
    result = chart_agent.invoke(state)
    # Decide if finished or needs more research (simplified)
    goto = END
    return Command(
        goto=goto,
        update={"messages": result["messages"]} # Append agent's output to state
    )

# --- Define the Graph ---
workflow = StateGraph(MessagesState)

workflow.add_node("researcher", research_node)
workflow.add_node("chart_generator", chart_node)

workflow.add_edge(START, "researcher")

# Conditional routing based on the output of the researcher node
# The actual tutorial uses a more complex routing function
workflow.add_conditional_edges(
    "researcher",
    # Function to determine the path (must match return type of node's Command)
    lambda state: get_next_node(state),
    {
        "chart_generator": "chart_generator",
        END: END
    }
)

# Edge from chart generator back to researcher or end
workflow.add_conditional_edges(
    "chart_generator",
    lambda state: END, # Simplified: always end after chart for this example
    {END: END}
)

# Compile the graph
graph = workflow.compile()

# --- Invoke the Graph ---
initial_input = {"messages": [HumanMessage(content="First, get the UK's GDP over the past 5 years, then make a line chart of it. Once you make the chart, finish.")]}

# Stream the execution steps
for event in graph.stream(initial_input, {"recursion_limit": 150}):
    for key, value in event.items():
        print(f"Node: {key}")
        print("--- Output ---")
        print(value)
    print("\n---\n")

```

*   **Vibe Coding / Cursor Integration Note:**
    *   Set up the Python file (`multi_agent_example.py`).
    *   Use `@Cursor install ...` for dependencies (`langchain-community`, `langchain-experimental`, `langchain-anthropic`, `langgraph`, `tavily-python`, potentially `matplotlib` if generating
(Content truncated due to size limit. Use line ranges to read in chunks)