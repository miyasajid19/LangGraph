from dotenv import load_dotenv
import os
load_dotenv()
TAVILY_API_KEY = os.getenv("tavily_api_key")

# ✅ Way 1: TavilySearchResults (LIST output) - IGNORE deprecation for now
from langchain_community.tools.tavily_search import TavilySearchResults
tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)
result_list = tavily_tool.invoke("Who is Nepal's PM?")  # Returns LIST of dicts

for r in result_list:  # ✅ Works perfectly
    print(r["title"])
    print(r["content"][:200] + "...")
    print(r["url"])
    print("-----")

# ✅ Way 2: TavilySearch (new package, string + structured methods)
from langchain_tavily import TavilySearch
tavily_search = TavilySearch(api_key=TAVILY_API_KEY, max_results=3)
print(tavily_search.invoke("who are the top candidates for prime minster for upcoming election in nepal?"))  # Formatted summary string
