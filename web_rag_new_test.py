import os
import time
import json
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
import os
import json
import google.generativeai as genai
from perplexity import Perplexity
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import time

# ============================================
# CONFIGURATION - Set your API keys here
# ============================================
GEMINI_API_KEY = "GEMINI_API_KEY"
PERPLEXITY_API_KEY ="PERPLEXITY_API_KEY"
COHERE_API_KEY = "COHERE_API_KEY"

# Configure APIs
genai.configure(api_key=GEMINI_API_KEY)
perplexity_client = Perplexity(api_key=PERPLEXITY_API_KEY)

# ===========
# Initialize ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="business_rules_collection",
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)

# ---------------------------
# STEP 1: EXTRACT ACTIONABLE STATEMENTS
# ---------------------------

def extract_actionable_statements(requirement: str) -> List[str]:
    """
    Use Gemini to break down user requirements into discrete, actionable statements.
    """
    print("="*60)
    print("STEP 1: Extracting Actionable Statements")
    print("="*60)

    prompt = f"""
    Break down the following requirement into distinct, concise actionable statements.
    Each statement should be short and specific (no filler text).

    Requirement:
    {requirement}
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    statements = [line.strip("•- ").strip()
                  for line in response.text.split("\n") if line.strip()]

    for i, stmt in enumerate(statements, 1):
        print(f"[{i}] {stmt}")
    return statements

# ---------------------------
# STEP 2: FETCH BUSINESS RULES FROM WEB (Perplexity + Gemini)
# ---------------------------

def fetch_business_rules_from_web(actionable_statement: str) -> Dict:
    """
    Fetches relevant business rules from the web using Perplexity Search API,
    then refines the result using Gemini to extract clear, concise business rules.
    """
    print(f"\n→ Fetching business rules for: '{actionable_statement[:3]}...'")

    try:
        search_query = (
            f"Extract business rules, compliance standards, or policy guidelines "
            f"related to: {actionable_statement}. "
            f"Summarize into clear business rules."
        )

        # Query Perplexity Search API
        search_response = perplexity_client.search.create(
            query=search_query,
            max_results=5
        )

        combined_snippets = []
        citations = []

        for result in search_response.results:
            snippet = getattr(result, "snippet", "")
            if snippet:
                combined_snippets.append(snippet.strip())
            citations.append({
                "title": getattr(result, "title", ""),
                "url": getattr(result, "url", "")
            })

        if not combined_snippets:
            return {
                "actionable_statement": actionable_statement,
                "business_rules": "No business rules found online.",
                "citations": []
            }

        business_rules_text = "\n".join(combined_snippets)

        # Refine using Gemini
        refinement_prompt = f"""
        You are a business compliance expert.
        Below is content from multiple web sources.

        Extract and rewrite ONLY the relevant business rules
        related to "{actionable_statement}".
        Output as a clean numbered list of rules (no commentary).

        Content:
        {business_rules_text}
        """

        model = genai.GenerativeModel("gemini-2.5-flash")
        refined = model.generate_content(refinement_prompt)
        clean_rules = refined.text.strip()

        print("  ✓ Extracted business rules successfully.")
        return {
            "actionable_statement": actionable_statement,
            "business_rules": clean_rules,
            "citations": citations,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        print(f"  ✗ Error fetching from web: {e}")
        return {
            "actionable_statement": actionable_statement,
            "business_rules": f"Error: {str(e)}",
            "citations": [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

# ---------------------------
# STEP 3: PERFORM WEB RAG
# ---------------------------

def perform_web_rag(actionable_statements: List[str]) -> List[Dict]:
    print("\n" + "="*60)
    print("STEP 3: Performing Web RAG (Business Rules Extraction)")
    print("="*60)

    all_results = []
    for idx, stmt in enumerate(actionable_statements, 1):
        print(f"\n[{idx}/{len(actionable_statements)}] {stmt}")
        result = fetch_business_rules_from_web(stmt)
        all_results.append(result)

        print("\n--- Extracted Business Rules ---")
        print(result["business_rules"])
        print("--------------------------------")
        time.sleep(2)
    return all_results

# ---------------------------
# STEP 4: STORE IN CHROMADB
# ---------------------------

def store_in_chromadb(collection, web_rag_results: List[Dict]):
    """
    Stores Web RAG results in ChromaDB safely.
    """
    print("\n" + "="*60)
    print("STEP 4: Storing Business Rules in ChromaDB")
    print("="*60)

    documents, metadatas, ids = [], [], []

    for idx, result in enumerate(web_rag_results):
        actionable_statement = result.get("actionable_statement", "")
        business_rules = result.get("business_rules", "")
        citations = result.get("citations", [])

        citation_urls = [c.get("url", "") for c in citations if c.get("url")]
        citation_titles = [c.get("title", "") for c in citations if c.get("title")]

        doc_text = f"""
Actionable Statement: {actionable_statement}

Business Rules:
{business_rules}

Sources:
{chr(10).join([f'- {t}: {u}' for t, u in zip(citation_titles, citation_urls)])}
""".strip()

        documents.append(doc_text)
        metadatas.append({
            "actionable_statement": actionable_statement,
            "citation_urls": json.dumps(citation_urls),
            "citation_titles": json.dumps(citation_titles),
            "timestamp": result.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        })
        ids.append(f"doc_{idx}_{int(time.time())}")

    try:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"✓ Stored {len(documents)} documents in ChromaDB")
    except Exception as e:
        print(f"✗ Error storing in ChromaDB: {e}")
        raise

# ---------------------------
# STEP 5: RETRIEVE TOP 5 RULES BY SEMANTIC SIMILARITY
# ---------------------------

def get_top_business_rules(collection, query_text: str, n_results: int = 5):
    """
    Retrieve top N most semantically similar business rules with URLs.
    """
    print("\n" + "="*60)
    print(f"TOP {n_results} BUSINESS RULES FOR QUERY:")
    print(query_text)
    print("="*60)

    results = collection.query(
        query_texts=[query_text],
        n_results=min(n_results, collection.count())
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        score = 1 - dist
        urls = json.loads(meta.get("citation_urls", "[]"))
        titles = json.loads(meta.get("citation_titles", "[]"))

        print(f"\n[{i}] Similarity: {score:.4f}")
        print(f"Actionable Statement: {meta['actionable_statement'][:120]}")
        if urls:
            print("Sources:")
            for t, u in zip(titles[:2], urls[:2]):
                print(f"  - {t}: {u}")
        snippet = doc[:400].strip().replace("\n", " ")
        print(f"\nBusiness Rules Preview:\n{snippet}...\n")

# ---------------------------
# MAIN WORKFLOW
# ---------------------------

def main():
    user_requirement = """
    ABC Customer SabseBadaRupaiya wants to build a Fund Manager Application for use by its talented money manager and esteemed customers. The Fund Manager Application will allow customer login, capture customer details, risk profile, investment amount. It will maintain each customer's current portfolio and update on real-time basis.
The application will be used by the manager to pool customer investments, and aid the manager in decision-making (algorithmic suggestions) to invest in various options such as fixed deposits, stocks, and mutual funds, again on real-time basis. The application should analyze return on investment options automatically by checking the web e.g. FD interest rates across banks, stock prices and mutual funds NAVs.
The goal is to build an LLM-driven application-builder that can "quickly" build this application for SabseBadaRupaiya, who insists that this application should attempt to deliver 10 percent returns for customers. Once done, ABC wants to market this as a magic fintech app-builder and sell-as-service to multiple customers, with possibly higher returns targeted.
    """

    # Step 1: Extract actionable statements
    actionable_statements = extract_actionable_statements(user_requirement)

    # Step 2–3: Perform Web RAG
    web_rag_results = perform_web_rag(actionable_statements)

    # Step 4: Store in ChromaDB
    store_in_chromadb(collection, web_rag_results)

    # Step 5: Retrieve top 5 rules based on similarity
    get_top_business_rules(collection, user_requirement, n_results=5)

# ---------------------------
# RUN
# ---------------------------

if __name__ == "__main__":
    main()
