import argparse
from datetime import datetime as dt

from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

from ..core.loader import DataLoader
from ..graph.graph import RouterGraph
from ..utils.logger import JSONLLogger
from ..retrieval.enricher import QueryEnricher
from ..retrieval.retrieval import AdaptiveRetriever
from ..tools.tools import build_registry
from ..utils.file import write_to_file
from ..spec.loader import load_graph_config, load_retrieval_config

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()

    # Read in test name 
    time = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser.add_argument("--test_name", default=time)
    args = parser.parse_args()

    log_path = f"logs/ragcv_headless_log_{args.test_name}.jsonl"

    loader = DataLoader(data_path="data_real/", db_path="data/db.faiss")
    embeddings = OpenAIEmbeddings()
    logger = JSONLLogger(log_path=log_path)

    # Construct vectorstore (either done from local save or re-generated if new data is present)
    documents = loader.get_documents()
    vectorstore = loader.load_vectorstore(embeddings=embeddings)
            
    # Setting up RAG
    retrieval_cfg = load_retrieval_config(path="config/retrieval.yml")
    
    retriever = AdaptiveRetriever(
        vectorstore=vectorstore, 
        documents=documents,
        config=retrieval_cfg,
        logger=logger
    )

    enricher = QueryEnricher(retriever=retriever,logger=logger)

    # Set up the tools to pass to agents
    tools_list = build_registry(test_name=args.test_name)

    # Construct agent objects using Pydantic type enforcing
    graph_cfg = load_graph_config(
        path="config/graph.yml",
        tools=tools_list,
        logger=logger,
        enricher=enricher
    )

    # Construct graph
    graph = RouterGraph(agents=graph_cfg, logger=logger)

    with open("input/query.txt", "r") as f:
        query = f.read()
    
    graph.draw() # Optional
    
    final_state = graph.invoke({
        'job_description' : HumanMessage(content=query),
    })

    write_to_file(final_state.get("document", "Error: No Document Found"), "output.txt")
    write_to_file(final_state.get("summary", "Error: No Summary Found"), "summary.txt")
    
    # Print console summary
    print("\n" + "="*60)
    print("JOB APPLICATION ASSISTANT - EXECUTION COMPLETE\n\n","="*60)
    print(logger.get_conversation_log())
    print("="*60,"\n\nCheck the 'output' directory for your tailored documents.")

if __name__ == "__main__":
    main()
