# This file will parse the query and process the request to the pipeline
# We will load documents and create a vector store, and a retriever
# We then set up the graph, passing in the retriever and the agents
# We will then run the graph, passing in the query
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from loader import DataLoader
from graph import RouterGraph
from agents import Agent
from tools import write_to_file
from dotenv import load_dotenv

def main():
    load_dotenv()

    # Load the data
    print("="*60)
    print("Loading data from files")
    print("="*60)
    loader = DataLoader(data_path="data", db_path="data/db.faiss")

    if not loader.db_similarity():
        vectorstore = loader.build_vectorstore()
    else:
        print("\n [Corpus unchanged since last build, loading existing vectorstore.]\n")
        vectorstore = FAISS.load_local(
                loader.db_path,
                OpenAIEmbeddings(),
                allow_dangerous_deserialization=True
            )

    retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2},
            )

    # Set up the tools
    tools_list = [
        write_to_file
    ]
    # Set up the graph
    agents = [
        {
            "name": "Router_Agent",
            "agent": Agent(prompt_path="prompts/router.txt"),
        },
        {
            "name": "CV_Agent",
            "agent": Agent(prompt_path="prompts/cv_agent.txt",
                retriever=retriever,
                tools=tools_list
                ),
        },
        {
            "name": "CL_Agent",
            "agent": Agent(prompt_path="prompts/cl_agent.txt",
                retriever=retriever,
                tools=tools_list
                ),
        },
        {
            "name": "Latex_Agent",
            "agent": Agent(prompt_path="prompts/latex_agent.txt",
                retriever=retriever,
                tools=tools_list
                ),
        },
        {
            "name": "Summary_Agent",
            "agent": Agent(prompt_path="prompts/summary_agent.txt"),
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
