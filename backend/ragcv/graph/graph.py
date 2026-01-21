# This file will assemble the graph logic from the agents, exposing an interface for the pipeline to use

from typing import TypedDict, List, Dict

from langchain_core.messages import  AnyMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from ..utils.logger import JSONLLogger
from ..factories.agent_factory import SpecialisedAgentFactory
from .state import RouterGraphState
from ..graph.node import AgentNodeWrapper

class RouterGraph:
    """
    Implements a general StateGraph object. Assumes that agents are arranged in order of rank.
    - Takes in a dict of agents and adds regular edges between agents with a difference in rank of one
    - If a conditional link is specified the graph will add a conditional edge between the link given 
        and all other nodes in the next ranks
    """
    def __init__(self, agents, logger: JSONLLogger):
        self.agents = agents
        self.logger = logger

        self.max_rank = max(agent['rank'] for agent in self.agents)
        self.min_rank = min(agent['rank'] for agent in self.agents)

        self.agent_map = {agent['name']: agent for agent in self.agents} | {'END': END}
        self.hierarchy = {
            rank: {'names': [], 'nodes': []}
            for rank in range(self.min_rank, self.max_rank +1)
        }
        self.agent_names = [agent["name"] for agent in agents]

        self.graph = StateGraph(RouterGraphState)

        for agent in self.agents:
            rank = agent['rank']

            self.hierarchy[rank]['names'].append(agent['name'])
            self.hierarchy[rank]['nodes'].append(agent['node'])
            self.graph.add_node(agent['name'], agent['node'])

        self.graph.set_entry_point(self.hierarchy[self.min_rank]['names'][0])

        self.add_edges()
        memory = MemorySaver()    

        self.graph = self.graph.compile(checkpointer=memory)
        self.config = {"configurable": {"thread_id": "default_thread"}}

    def add_edges(self):
        """Add regular edges between adjacent nodes and conditional edges where specified"""
        for rank in self.hierarchy:
            for source_node in self.hierarchy[rank]['names']:
                agent_data = self.agent_map[source_node]

                if 'conditional_links' in agent_data:
                    self._add_conditional_edges(source_node, agent_data, rank)
                elif 'hard_links' in agent_data:
                    self._add_regular_edges(source_node, rank, hard_link=True)
                else:
                    self._add_regular_edges(source_node, rank)
                    
    def _get_target_dict(self, rank):
        """Helper function to get target dictionary for routing to next rank or 'END'"""
        if rank < self.max_rank:
            targets = self.hierarchy[rank +1]
            return {target_node: target_node for target_node in targets['names']}
        else: 
            return {"END": END}

    def _add_conditional_edges(self, source_node, agent_data, rank):
        """Add conditional edges for a source node"""
        cond_links = agent_data['conditional_links']

        if any(item not in self.agent_map for item in cond_links):
            raise RuntimeError('Conditional link invalid; No agent to fulfill selection')
        
        target_dict = self._get_target_dict(rank)

        routing_dict = {
            cond_node: cond_node for cond_node in cond_links
        } | target_dict

        routing_dict |= self._get_target_dict(rank)

        self.graph.add_conditional_edges(
            source_node,
            self.route,
            routing_dict
        )
    
    def _add_regular_edges(self, source_node, rank, hard_link=False):
        """Add regular edges for a source node"""
        agent_data = self.agent_map[source_node]
        
        if hard_link:
            hard_links = agent_data['hard_links']
            if any(node not in self.agent_map for node in hard_links):
                raise RuntimeError(f'Invalid hard link in {source_node}; No agent to fulfill selection')
            for target_node in hard_links:
                self.graph.add_edge(source_node, target_node)
            return

        if rank < self.max_rank and hard_link == False:
            target_dict = self._get_target_dict(rank)
            for target_node in target_dict.keys():
                self.graph.add_edge(source_node, target_node)
        else:  
            self.graph.add_edge(source_node, END)

    def route(self, state: RouterGraphState):

        router_input = state['latest_message']

        task = state.get('task', None)
        status = router_input.get('status', None) 

        route_map = {
                'Cover Letter': 'CL_Task_Agent',
                'CV': 'CV_Task_Agent'
            }

        pathway = route_map[task]

        if status:
            pathway = 'END' if status == 'PASS' else route_map[task]

        if pathway == "END":
            self.logger.log_conversation()
            return "END"

        # Handle invalid pathway:
        if pathway not in self.graph.nodes.keys():
            print(f"[Error: Invalid pathway: {pathway}]")
            return "END"

        return pathway

    def draw(self):
        """Creates a mermaid image of the graph and saves it"""
        self.graph.get_graph().draw_mermaid_png(output_file_path="img/graph.png")
        
    def invoke(self, message):
        output = self.graph.invoke(message, config=self.config)
        return output




