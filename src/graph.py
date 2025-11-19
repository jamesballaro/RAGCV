# This file will assemble the graph logic from the agents, exposing an interface for the pipeline to use
import os
import operator

from typing import TypedDict, List, Annotated

from langchain_core.messages import  AnyMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
class RouterGraphState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

class RouterGraph:
    def __init__(self, agents = []):
        self.agents = agents
        self.agent_names = [agent["name"] for agent in agents]
        self.execution_log = []

        graph = StateGraph(RouterGraphState)

        for agent in agents:
            original_agent = agent["agent"]
            setattr(original_agent, "name", agent["name"])
            agent["agent"] = self._wrap_agent(original_agent, agent["name"])
            graph.add_node(agent["name"], agent["agent"])

        graph.add_conditional_edges(
            agents[0]["name"],
            self.route,
            {
                agent["name"]: agent["name"] for agent in agents[1:]
            } | {"END": END}
        )

        for agent in agents[1:]:
            graph.add_edge(agent["name"], agents[0]["name"])

        graph.set_entry_point(agents[0]["name"])
        memory = MemorySaver()

        self.graph = graph.compile(checkpointer=memory)

    def _wrap_agent(self, agent, agent_name):
        """Wrapper to log agent executions"""
        def wrapped_agent(state):
            result = agent(state)
            if result.get("messages"):
                last_msg = result["messages"][-1]
                self.execution_log.append({
                    "agent": agent_name,
                    "message": last_msg.content
                })
            return result
        return wrapped_agent

    def  route(self, state: RouterGraphState):
        last_message = state["messages"][-1]
        pathway =  last_message.content

        # Handle invalid pathway:
        if pathway not in self.agent_names and pathway != "END":
            print(f"[Error: Invalid pathway: {pathway}]")
            return "END"

        runs = {entry['name']: 0 for entry in self.agents}
        for entry in self.execution_log:
            runs[entry['agent']] += 1

        # TODO: Fix this hacky solution to prevent infinite loops
        if runs.get(pathway, 0) >= 2:
            print(f"[Error: Detected potential infinite loop with pathway: {pathway}]\n \t[Loop forcibly terminated.]")
            return "END"

        return pathway

    def get_execution_summary(self):
        if not self.execution_log:
            return "No agents were executed."

        summary = "Execution Summary:\n" + "="*50 + "\n"
        for entry in self.execution_log:
            summary += f"\n[{entry['agent']}]\n{entry['message']}\n"
        return summary

    def invoke(self, message, config = None):
        self.execution_log = []
        if config is None:
            config={"configurable": {"thread_id": "default_thread"}}
        output = self.graph.invoke(message, config=config)
        return output




