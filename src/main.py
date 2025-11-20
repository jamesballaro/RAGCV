# This file will parse the query and process the request to the pipeline
# We will load documents and create a vector store, and a retriever
# We then set up the graph, passing in the retriever and the agents
# We will then run the graph, passing in the query
import argparse
from datetime import datetime as dt

from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from loader import DataLoader
from graph import RouterGraph
from agents import Agent
from tools import write_to_file
from dotenv import load_dotenv
from logger import AgentJSONLLogger
from utils.retrieval import AdaptiveRetriever

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()

    time = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser.add_argument("--test_name", default=time)
    args = parser.parse_args()

    # Load the data
    print("="*60)
    print("Loading data from files")
    print("="*60)
    loader = DataLoader(data_path="data", db_path="data/db.faiss")
    embeddings = OpenAIEmbeddings()

    if not loader.db_similarity():
        vectorstore = loader.build_vectorstore(embeddings=embeddings)
    else:
        print("\n [Corpus unchanged since last build, loading existing vectorstore.]\n")
        vectorstore = FAISS.load_local(
                loader.db_path,
                embeddings,
                allow_dangerous_deserialization=True
            )

    cv_retriever = AdaptiveRetriever(vectorstore=vectorstore, embeddings=embeddings, agent_type="cv")
    cl_retriever = AdaptiveRetriever(vectorstore=vectorstore, embeddings=embeddings, agent_type="cl")

    log_path = f"logs/agent_runs_{args.test_name}.jsonl"
    agent_logger = AgentJSONLLogger(log_path=log_path)

    # Set up the tools
    tools_list = [
        write_to_file
    ]
    # Set up the graph
    agents = [
        {
            "name": "Router_Agent",
            "agent": Agent(
                prompt_path="prompts/router.txt",
                logger=agent_logger,
            ),
        },
        {
            "name": "CV_Agent",
            "agent": Agent(
                prompt_path="prompts/cv_agent.txt",
                retriever=cv_retriever,
                tools=tools_list,
                logger=agent_logger,
                agent_type="cv",
            ),
        },
        {
            "name": "CL_Agent",
            "agent": Agent(
                prompt_path="prompts/cl_agent.txt",
                retriever=cl_retriever,
                tools=tools_list,
                logger=agent_logger,
                agent_type="cl",
            ),
        },
        # {
        #     "name": "Latex_Agent",
        #     "agent": Agent(
        #         prompt_path="prompts/latex_agent.txt",
        #         retriever=latex_retriever,
        #         tools=tools_list,
        #         logger=agent_logger,
        #         agent_type="latex",
        #     ),
        # },
        {
            "name": "Summary_Agent",
            "agent": Agent(
                prompt_path="prompts/summary_agent.txt",
                tools=tools_list,
                logger=agent_logger,
            ),
        }
    ]

    graph = RouterGraph(agents=agents)

    with open("input/query.txt", "r") as f:
        query = f.read()

    message = {'messages' : [HumanMessage(content=query)]}
    output = graph.invoke(message)

    # Print console summary (requirement 1)
    print("\n" + "="*60)
    print("JOB APPLICATION ASSISTANT - EXECUTION COMPLETE")
    print("="*60)
    print(graph.get_execution_summary())
    print("="*60)
    print("\nCheck the 'output' directory for your tailored documents.")

if __name__ == "__main__":
    main()
