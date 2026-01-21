from typing import Dict, Any, Sequence, List

from ..workflows.processor import StateProcessor
from ..core.agent import Agent

class AgentNodeWrapper:
    def __init__(
            self,
            agent: Agent,
            agent_name: str,
            logger
        ):
        self.agent = agent
        self.agent_name = agent_name
        self.logger = logger

        self.processor = StateProcessor(agent_name)
    
    def __call__(self, state: Dict[Any,Any]):
        print(f'{"="*60}\nAgent called: {self.agent_name}\n{"="*60}')
        passed_validation = False
        max_retries = 3
        retry_count = 0

        input_data = self.processor.prepare_input(state) 
        output_state = None

        while not passed_validation and retry_count < max_retries:
            try: 
                output_data, tool_result  = self.agent.invoke(input_data)

                tool_logs = self.process_tool_call(tool_result, self.agent.tools)
            
                self.logger.log_agent_invocation(
                    agent_name=self.agent_name,
                    input_message=input_data,
                    output_message=output_data,
                    tool_logs=tool_logs if tool_logs else {}
                )

                output_state = self.processor.prepare_output(output_data, state) 

                if output_state is not None:
                    passed_validation = True

            except Exception as e:
                retry_count += 1
                self.logger.log_agent_error(agent_name=self.agent_name, error_message=str(e))
                print(f"{'='*60}\nError whilst invoking agent {self.agent_name}: {e}\n{'='*60}")
                print(f"{'='*60}\nAttempting re-run of {self.agent_name}\n{'='*60}")

                if retry_count >= max_retries:
                    raise RuntimeError(f"Agent {self.agent_name} failed after {max_retries} retries: {e}") from e
        
        return output_state
    
    def process_tool_call(self,
        result: Any,
        tools: Sequence[Any],
    ):
        if result and hasattr(result, "tool_calls") and result.tool_calls:
            print(f"Processing tool calls from {self.agent_name}")

            tool_logs: List[Dict[str, Any]] = []
            tool_map = {t.name: t for t in tools}

            for call in result.tool_calls:
                tool_name = call["name"]
                tool_input = call["args"]

                print(f'{"-"*60}\nTool called: {call["name"]}()\n{"-"*60}')

                matching_tool = tool_map.get(tool_name)

                if not matching_tool:
                    raise ValueError(f"Tool '{tool_name}' not found in agent's tool list.")
                
                try:
                    tool_output = matching_tool.invoke(tool_input)
                except Exception as e:
                    tool_output = f"[Error executing tool: {e}]"

                tool_logs.append(
                    {"tool_name": tool_name, "args": tool_input, "output": str(tool_output)}
                )

            return tool_logs
        else:
            return {}