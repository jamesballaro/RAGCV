import time
import traceback

from typing import Dict, Any, Sequence, List
import time
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from ..workflows.processor import StateProcessor
from ..core.agent import Agent
class LatencyMonitorCallback(BaseCallbackHandler):
    def __init__(self):
        self.start_time = 0.0
        self.first_token_time = 0.0
        self.token_count = 0
        self.metrics = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()

    def on_llm_new_token(self, token: str, **kwargs):
        if self.token_count == 0:
            self.first_token_time = time.time()
            self.metrics["ttft"] = self.first_token_time - self.start_time
        self.token_count += 1

    def on_llm_end(self, response: LLMResult, **kwargs):
        end_time = time.time()
        total_time = end_time - self.start_time

        self.metrics["total_time"] = total_time
        self.metrics["token_count"] = self.token_count

        if self.first_token_time > 0:
            gen_time = end_time - self.first_token_time
            self.metrics["generation_time"] = gen_time
            self.metrics["tokens_per_second"] = (
                self.token_count / gen_time if gen_time > 0 else 0.0
            )
        else:
            self.metrics["generation_time"] = None
            self.metrics["tokens_per_second"] = None

class AgentNodeWrapper:
    def __init__(
            self,
            agent: Agent,
            agent_name: str,
            logger,
            enricher
        ):
        self.agent = agent
        self.agent_name = agent_name
        self.logger = logger
        self.enricher = enricher
        self.processor = StateProcessor(agent_name, enricher)
    
    def __call__(self, state: Dict[Any,Any]):
        print(f'{"="*60}\nAgent called: {self.agent_name}\n{"="*60}')
        passed_validation = False
        max_retries = 3
        retry_count = 0

        print(f'\t* Preparing input: {self.agent_name}')
        input_data = self.processor.prepare_input(state) 
        output_state = None

        while not passed_validation and retry_count < max_retries:
            try: 
                print(f'\t* Invoking agent: {self.agent_name}')
                
                start_time = time.time()
                callback = LatencyMonitorCallback()
                
                output_data, tool_result = self.agent.invoke(
                    input_data,
                    callbacks=lambda: callback
                )
                elapsed_time = time.time() - start_time
                
                latency_metrics = callback.metrics

                print(f"\t* Agent invocation took {elapsed_time:.2f} seconds")

                tool_logs = self.process_tool_call(tool_result, self.agent.tools)
            
                self.logger.log_agent_invocation(
                    agent_name=self.agent_name,
                    input_message=input_data,
                    output_message=output_data,
                    tool_logs=tool_logs if tool_logs else {},
                    latency_metrics=latency_metrics
                )
                print(f'\t* Preparing output: {self.agent_name}\n')

                output_state = self.processor.prepare_output(output_data, state) 

                if output_state is not None:
                    passed_validation = True

            except Exception as e:
                retry_count += 1
                tb_str = traceback.format_exc()
                self.logger.log_agent_error(
                    agent_name=self.agent_name, 
                    error_message=str(e),
                    traceback=tb_str
                )
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