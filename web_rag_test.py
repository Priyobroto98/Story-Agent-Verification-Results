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

# ============================================
# STEP 1: Convert User Requirements to Actionable Statements
# ============================================
def convert_to_actionable_statements(user_requirement: str) -> List[str]:
    """
    Converts user requirements into actionable statements using Gemini 2.5 Flash.
    """
    print("\n" + "="*60)
    print("STEP 1: Converting User Requirement to Actionable Statements")
    print("="*60)
    print(f"\nOriginal User Requirement:\n{user_requirement}\n")
    
    prompt = f"""
    You are an expert business analyst. Convert the following user requirement into clear, 
    specific, actionable statements. Each statement should be concise and focused on a single action or rule.
    
    IMPORTANT: Maintain the originality and intent of the user requirement. Do not add interpretation 
    or assumptions beyond what is explicitly stated.
    
    Format your response as a JSON array of strings, where each string is one actionable statement.
    
    User Requirement:
    {user_requirement}
    
    Return only the JSON array, nothing else.
    """
    
    try:
        # Use Gemini for conversion
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(prompt)
        
        print(f"Gemini Response:\n{response.text}\n")
        
        response_text = response.text.strip()
        
        # Remove markdown code block markers if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        actionable_statements = json.loads(response_text)
        
        print(f"✓ Successfully converted to {len(actionable_statements)} actionable statements:")
        for idx, statement in enumerate(actionable_statements, 1):
            print(f"  {idx}. {statement}")
        
        return actionable_statements
        
    except Exception as e:
        print(f"✗ Error in converting to actionable statements: {e}")
        print(f"Response text was: {response_text[:200]}...")
        raise

# ============================================
# STEP 2: Perform Web RAG using Perplexity Search API
# ============================================
# def fetch_business_rules_from_web(actionable_statement: str) -> Dict:
#     print(f"\n  → Searching web for: '{actionable_statement[:60]}...'")
    
#     try:
#         search_query = f"business rules best practices standards guidelines: {actionable_statement}"
        
#         search_response = perplexity_client.search.create(
#             query=search_query,
#             max_results=10,
#             max_tokens_per_page=2048
#         )
        
#         citations = []
#         combined_content = []
        
#         for idx, result in enumerate(search_response.results, 1):
#             citation = {
#                 'title': getattr(result, 'title', ''),
#                 'url': getattr(result, 'url', ''),
#                 'snippet': getattr(result, 'snippet', ''),
#                 'date': getattr(result, 'date', ''),
#                 'rank': idx
#             }
#             citations.append(citation)
            
#             if getattr(result, 'snippet', ''):
#                 combined_content.append(f"[{idx}] {result.title}:\n{result.snippet}\n")
        
#         business_rules_text = "\n".join(combined_content)
        
#         result = {
#             "actionable_statement": actionable_statement,
#             "business_rules": business_rules_text,
#             "citations": citations,
#             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#             "total_results": len(citations)
#         }
        
#         print(f"  ✓ Found {len(citations)} search results")
#         print(f"  ✓ Retrieved {len(business_rules_text)} characters of content")
        
#         return result
        
#     except Exception as e:
#         print(f"  ✗ Error fetching from web: {e}")
#         return {
#             "actionable_statement": actionable_statement,
#             "business_rules": f"Error: {str(e)}",
#             "citations": [],
#             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#             "total_results": 0
#         }
def fetch_business_rules_from_web(actionable_statement: str) -> Dict:
    """
    Fetches relevant business rules text from the web using Perplexity Search API.
    Focused purely on extracting usable business rules or policy guidance.
    """
    print(f"\n→ Fetching business rules for: '{actionable_statement[:80]}...'")

    try:
        # Make search query precise
        search_query = (
            f"Extract business rules, compliance standards, or policy guidelines "
            f"related to: {actionable_statement}. "
            f"Summarize into clear business rules in plain text."
        )

        # Query Perplexity Search API
        search_response = perplexity_client.search.create(
            query=search_query,
            max_results=3,
            max_tokens_per_page=2048
        )

        # Combine relevant snippets
        combined_snippets = []
        for result in search_response.results:
            if getattr(result, "snippet", ""):
                combined_snippets.append(result.snippet.strip())

        if not combined_snippets:
            print("  ⚠ No business rule snippets found.")
            return {
                "actionable_statement": actionable_statement,
                "business_rules": "No business rules found online.",
            }

        # Combine and clean text
        business_rules_text = "\n".join(combined_snippets)

        # Use Gemini (optional) to extract only the 'rules' portion cleanly
        refinement_prompt = f"""
        You are a business compliance expert.
        Below is raw content gathered from multiple web sources.

        Please extract and rewrite ONLY the relevant business rules,
        policies, or best practices that directly relate to:
        "{actionable_statement}"

        Output should be a clean numbered list of business rules, no commentary.

        Content:
        {business_rules_text}
        """

        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        refined = model.generate_content(refinement_prompt)

        clean_rules = refined.text.strip()

        print(f"  ✓ Extracted business rules successfully.")

        return {
            "actionable_statement": actionable_statement,
            "business_rules": clean_rules,
        }

    except Exception as e:
        print(f"  ✗ Error fetching from web: {e}")
        return {
            "actionable_statement": actionable_statement,
            "business_rules": f"Error: {str(e)}",
        }
def perform_web_rag(actionable_statements: List[str]) -> List[Dict]:
    """
    Performs simplified Web RAG — returns and prints only the extracted business rules.
    """
    print("\n" + "="*60)
    print("STEP 2: Fetching Business Rules (Simplified Output)")
    print("="*60)

    all_results = []

    for idx, statement in enumerate(actionable_statements, 1):
        print(f"\n[{idx}/{len(actionable_statements)}] {statement}")
        result = fetch_business_rules_from_web(statement)
        all_results.append(result)

        print("\n--- Extracted Business Rules ---")
        print(result["business_rules"])
        print("--------------------------------")

        if idx < len(actionable_statements):
            time.sleep(2)

    print(f"\n✓ Completed fetching business rules for {len(all_results)} actionable statements.\n")
    return all_results


# def perform_web_rag(actionable_statements: List[str]) -> List[Dict]:
#     print("\n" + "="*60)
#     print("STEP 2: Performing Web RAG with Perplexity Search API")
#     print("="*60)
    
#     all_results = []
    
#     for idx, statement in enumerate(actionable_statements, 1):
#         print(f"\n[{idx}/{len(actionable_statements)}] Processing statement...")
#         result = fetch_business_rules_from_web(statement)
#         all_results.append(result)
        
#         if result['citations']:
#             print(f"\n  Top Citations:")
#             for cit_idx, citation in enumerate(result['citations'][:5], 1):
#                 print(f"    [{cit_idx}] {citation['title']}")
#                 print(f"        URL: {citation['url']}")
#                 if citation.get('date'):
#                     print(f"        Date: {citation['date']}")
#                 if citation.get('snippet'):
#                     snippet_preview = citation['snippet'][:100].replace('\n', ' ')
#                     print(f"        Snippet: {snippet_preview}...")
        
#         if idx < len(actionable_statements):
#             print(f"  ⏱ Waiting 2 seconds before next request...")
#             time.sleep(2)
    
#     print(f"\n✓ Completed Web RAG for all {len(actionable_statements)} statements")
#     print(f"✓ Total documents retrieved: {sum(r['total_results'] for r in all_results)}")
    
#     return all_results

# ============================================
# STEP 3: Store in ChromaDB with Cohere Embeddings
# ============================================
def initialize_chromadb_with_cohere(persist_directory: str = "./chroma_db"):
    print("\n" + "="*60)
    print("STEP 3: Initializing ChromaDB with Cohere Embeddings")
    print("="*60)
    print(f"\nPersist Directory: {persist_directory}")
    
    try:
        cohere_ef = embedding_functions.CohereEmbeddingFunction(
            api_key=COHERE_API_KEY,
            model_name="embed-english-v3.0"
        )
        
        print("✓ Cohere embedding function created")
        
        client = chromadb.PersistentClient(path=persist_directory)
        print(f"✓ ChromaDB persistent client initialized")
        
        collection = client.get_or_create_collection(
            name="business_rules_collection",
            embedding_function=cohere_ef,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✓ Collection 'business_rules_collection' ready")
        print(f"  Current document count: {collection.count()}")
        
        return client, collection
        
    except Exception as e:
        print(f"✗ Error initializing ChromaDB: {e}")
        raise

# def store_in_chromadb(collection, web_rag_results: List[Dict]):
#     print("\n" + "="*60)
#     print("STEP 4: Storing Business Rules in ChromaDB")
#     print("="*60)
    
#     documents = []
#     metadatas = []
#     ids = []
    
#     for idx, result in enumerate(web_rag_results):
#         citation_text = ""
#         if result['citations']:
#             citation_text = "\n\nSources:\n"
#             for cit in result['citations']:
#                 citation_text += f"[{cit['rank']}] {cit['title']}\n"
#                 citation_text += f"    URL: {cit['url']}\n"
#                 if cit.get('date'):
#                     citation_text += f"    Date: {cit['date']}\n"
        
#         doc_text = f"""
# Actionable Statement: {result['actionable_statement']}

# Business Rules and Documentation:
# {result['business_rules']}
# {citation_text}
#         """.strip()
        
#         documents.append(doc_text)
        
#         citation_urls = [c['url'] for c in result['citations']]
#         citation_titles = [c['title'] for c in result['citations']]
        
#         metadata = {
#             "actionable_statement": result['actionable_statement'][:500],
#             "timestamp": result['timestamp'],
#             "citation_count": result['total_results'],
#             "has_citations": result['total_results'] > 0,
#             "citation_urls": json.dumps(citation_urls[:10]),
#             "citation_titles": json.dumps(citation_titles[:10])
#         }
#         metadatas.append(metadata)
        
#         ids.append(f"doc_{idx}_{int(time.time())}")
    
#     try:
#         collection.add(
#             documents=documents,
#             metadatas=metadatas,
#             ids=ids
#         )
        
#         print(f"✓ Successfully stored {len(documents)} documents in ChromaDB")
#         print(f"  Total documents in collection: {collection.count()}")
        
#         print("\nStored Documents Summary:")
#         for idx, (doc_id, metadata) in enumerate(zip(ids, metadatas), 1):
#             print(f"\n  [{idx}] ID: {doc_id}")
#             print(f"      Statement: {metadata['actionable_statement'][:80]}...")
#             print(f"      Citations: {metadata['citation_count']}")
            
#             if metadata['citation_count'] > 0:
#                 urls = json.loads(metadata['citation_urls'])
#                 titles = json.loads(metadata['citation_titles'])
#                 print(f"      Top Source: {titles[:1]}...")
#                 print(f"      URL: {urls[:1]}...")
        
#     except Exception as e:
#         print(f"✗ Error storing in ChromaDB: {e}")
#         import traceback
#         traceback.print_exc()
#         raise

def store_in_chromadb(collection, web_rag_results: List[Dict]):
    """
    Stores Web RAG results in ChromaDB safely.
    Handles both detailed and simplified results.
    """
    print("\n" + "="*60)
    print("STEP 4: Storing Business Rules in ChromaDB")
    print("="*60)

    documents = []
    metadatas = []
    ids = []

    for idx, result in enumerate(web_rag_results):
        actionable_statement = result.get("actionable_statement", "N/A")
        business_rules = result.get("business_rules", "No rules found.")

        # Handle missing citations gracefully
        citations = result.get("citations", [])
        citation_text = ""
        citation_urls = []
        citation_titles = []

        if citations:
            citation_text = "\n\nSources:\n"
            for cit in citations:
                title = cit.get("title", "Untitled")
                url = cit.get("url", "")
                citation_text += f"- {title}\n  {url}\n"
                citation_urls.append(url)
                citation_titles.append(title)

        # Combine into full text
        doc_text = f"""
Actionable Statement: {actionable_statement}

Business Rules:
{business_rules}
{citation_text}
""".strip()

        documents.append(doc_text)

        metadata = {
            "actionable_statement": actionable_statement[:500],
            "timestamp": result.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            "citation_count": len(citations),
            "citation_urls": json.dumps(citation_urls[:10]),
            "citation_titles": json.dumps(citation_titles[:10]),
        }

        metadatas.append(metadata)
        ids.append(f"doc_{idx}_{int(time.time())}")

    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"✓ Successfully stored {len(documents)} documents in ChromaDB")
        print(f"  Total documents in collection: {collection.count()}")

    except Exception as e:
        print(f"✗ Error storing in ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================
# STEP 5: Perform Similarity Search
# ============================================
def perform_similarity_search(collection, actionable_statements: List[str], n_results: int = 3):
    print("\n" + "="*60)
    print("STEP 5: Performing Similarity Search")
    print("="*60)
    
    for idx, statement in enumerate(actionable_statements, 1):
        print(f"\n[{idx}/{len(actionable_statements)}] Searching for: '{statement[:60]}...'")
        
        try:
            results = collection.query(
                query_texts=[statement],
                n_results=min(n_results, collection.count())
            )
            
            print(f"  ✓ Found {len(results['ids'])} similar documents")
            
            for result_idx, (doc_id, distance, metadata) in enumerate(
                zip(results['ids'], results['distances'], results['metadatas']), 1
            ):
                print(f"\n  Result {result_idx}:")
                print(f"    ID: {doc_id}")
                print(f"    Similarity Score: {1 - distance:.4f}")
                print(f"    Statement: {metadata['actionable_statement'][:100]}...")
                print(f"    Citation Count: {metadata['citation_count']}")
                
                if metadata.get('citation_urls') and metadata.get('citation_titles'):
                    urls = json.loads(metadata['citation_urls'])
                    titles = json.loads(metadata['citation_titles'])
                    if urls and titles:
                        print(f"    Top Citation: {titles[:1]}...")
                        print(f"    URL: {urls[:1]}...")
                
        except Exception as e:
            print(f"  ✗ Error in similarity search: {e}")
def get_top_business_rules(collection, query_text: str, n_results: int = 5):
    """
    Retrieve the top N most semantically similar business rules from ChromaDB
    along with their source URLs for reliability.
    """
    print("\n" + "="*60)
    print(f"TOP {n_results} BUSINESS RULES FOR QUERY:")
    print(query_text)
    print("="*60)

    try:
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

            # Show only the top 1–2 URLs for brevity
            if urls:
                print("Sources:")
                for u, t in zip(urls[:2], titles[:2]):
                    print(f"  - {t}: {u}")

            # Display a short preview of the document content
            snippet = doc[:400].strip().replace("\n", " ")
            print(f"\nBusiness Rules Preview:\n{snippet}...\n")

    except Exception as e:
        print(f"✗ Error retrieving business rules: {e}")

# ============================================
# MAIN WORKFLOW
# ============================================
def main():
    print("\n" + "="*70)
    print(" USER REQUIREMENT TO WEB RAG WORKFLOW")
    print(" Using Perplexity Search API for Document Retrieval")
    print("="*70)
    
    user_requirement = """
    ABC Customer SabseBadaRupaiya wants to build a Fund Manager Application for use by its talented money manager and esteemed customers. The Fund Manager Application will allow customer login, capture customer details, risk profile, investment amount. It will maintain each customer's current portfolio and update on real-time basis.
The application will be used by the manager to pool customer investments, and aid the manager in decision-making (algorithmic suggestions) to invest in various options such as fixed deposits, stocks, and mutual funds, again on real-time basis. The application should analyze return on investment options automatically by checking the web e.g. FD interest rates across banks, stock prices and mutual funds NAVs.
The goal is to build an LLM-driven application-builder that can "quickly" build this application for SabseBadaRupaiya, who insists that this application should attempt to deliver 10 percent returns for customers. Once done, ABC wants to market this as a magic fintech app-builder and sell-as-service to multiple customers, with possibly higher returns targeted.
    """
    
    try:
        actionable_statements = convert_to_actionable_statements(user_requirement)
        web_rag_results = perform_web_rag(actionable_statements)
        client, collection = initialize_chromadb_with_cohere()
        store_in_chromadb(collection, web_rag_results)
        perform_similarity_search(collection, actionable_statements)
        
        print("\n" + "="*70)
        print(" WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\n✓ All steps executed successfully!")
        print(f"✓ {len(actionable_statements)} actionable statements processed")
        print(f"✓ {sum(r['total_results'] for r in web_rag_results)} total documents retrieved")
        print(f"✓ {len(web_rag_results)} documents stored in ChromaDB")
        print(f"✓ ChromaDB collection contains {collection.count()} total documents")
        print("\nYou can now perform similarity searches on the stored business rules.",get_top_business_rules(collection, query_text, n_results=5))
    except Exception as e:
        print(f"\n✗ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
