# This file will parse the query and process the request to the pipeline
# We will load documents and create a vector store, and a retriever
# We then set up the graph, passing in the retriever and the agents
# We will then run the graph, passing in the query
import argparse
from datetime import datetime as dt

from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv


from .loader import DataLoader
from .graph import RouterGraph
from .tools.tools import build_registry
from .utils.logger import JSONLLogger
from .retrieval.enricher import QueryEnricher
from .retrieval.retrieval import AdaptiveRetriever
from .spec.loader import load_from_yaml

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()

    # Read in test name 
    time = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser.add_argument("--test_name", default=time)
    args = parser.parse_args()

    loader = DataLoader(data_path="data", db_path="data/db.faiss")
    embeddings = OpenAIEmbeddings()

    # Construct vectorstore (either done from local save or re-generated if new data is present)
    vectorstore = loader.load_vectorstore(embeddings=embeddings)
        
    # Logging
    log_path = f"logs/agent_runs_{args.test_name}.jsonl"
    logger = JSONLLogger(log_path=log_path)

    # Setting up RAG
    retriever = AdaptiveRetriever(vectorstore=vectorstore, embeddings=embeddings)
    enricher = QueryEnricher(retriever=retriever,logger=logger)

    # Set up the tools to pass to agents
    tools_list = build_registry(test_name=args.test_name)

    # Construct agent objects using Pydantic type enforcing
    agents = load_from_yaml(
        path="config/graph.yml",
        tools=tools_list,
        logger=logger,
    )

    # Construct graph
    graph = RouterGraph(agents=agents, logger=logger)

    with open("input/query.txt", "r") as f:
        query = f.read()

    message = enricher.enrich_with_context(query)

    graph.draw() # Optional
    
    graph.invoke({'messages' : [HumanMessage(content=message)]})

    # Print console summary (requirement 1)
    print("\n" + "="*60)
    print("JOB APPLICATION ASSISTANT - EXECUTION COMPLETE\n\n","="*60)
    print(graph.get_execution_summary())
    print("="*60,"\n\nCheck the 'output' directory for your tailored documents.")

if __name__ == "__main__":
    main()
