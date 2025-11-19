from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
import arxiv
import os
import requests
from pypdf import PdfReader
import chromadb
from google import genai

DB_PATH = "vector_store"
EMBEDDING_MODEL = "text-embedding-004"
DOWNLOAD_DIR = "papers"
def download_to_pdf(url: str, filename: str) -> str:
    print(f"\n[tool] Attempting to download: {filename} at {url}")
    try:
        clean_filename = os.path.basename(filename)
        if "arxiv.org" in url and not url.endswith(".pdf"):
            url = url.replace("v1", "")
            if not url.endswith(".pdf"):
                url += ".pdf"
            if not clean_filename.endswith('.pdf'):
                clean_filename += ".pdf"

        folder = os.path.join(DOWNLOAD_DIR, clean_filename)
        if not os.path.exists(DOWNLOAD_DIR):
            os.makedirs(DOWNLOAD_DIR)
            print(f"[DEBUG] Created folder: {DOWNLOAD_DIR}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers = headers, timeout = 30)
        if response.status_code == 200:
            with open(folder, 'wb') as f:
                f.write(response.content)
            print(f"File written to: {folder}")
            return f"success: saved {folder}"
        else:
            print(f"[DEBUG] Failed with status {response.status_code}")
            return f"[DEBUG] Failed with status {response.status_code}"
    except Exception as e:
        return f"Error downloading :( {str(e)}"

def search_arxiv_tool(query: str) -> str:
    """
    Searches Arxiv.org for research papers and returns titles and PDF links.
    Args:
        query: The topic to search for (e.g. "AGI", "LLM").
    """
    print(f"\n[Tool] Searching Arxiv for: {query}...")
    try:
        # Construct the client
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        results = []
        for r in client.results(search):
            # Arxiv provides a direct .pdf link attribute
            results.append(f"Title: {r.title}\nPDF_URL: {r.pdf_url}\nSummary: {r.summary[:100]}...\n")
        if not results:
            return "No papers found."
        return "\n".join(results)
    except Exception as e:
        return f"Arxiv search failed: {str(e)}"

def ingest_to_db() -> str:
    print(f"\n[Librarian] Starting the embedding process")
    files = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".pdf")]
    if not files:
        return f"There are no files to ingest"
    count = 0
    for f in files:
        path = os.path.join(DOWNLOAD_DIR, f)
        print(f"[Librarian] Processing file {f}")
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        ids = [f"{f}_chunk_{i}" for i in range(len(chunks))]
        # Call Google API to get embeddings for these chunks
        # Note: In production, batch this!
        print(f"   - Embedding {len(chunks)} chunks...")
        embeddings = []
        for chunk in chunks:
            resp = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=chunk
            )
            embeddings.append(resp.embeddings[0].values)

        # Add to Chroma
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"source": f}] * len(chunks)
        )
        count += 1

    return f"Successfully ingested {count} documents into the Vector Database."

search_agent = Agent(
    name="search_assistant",
    model = "gemini-2.0-flash",
    instruction="""You are a research assistant.
    1. search for at least 10 latest research papers.
    2. Find the direct pdf files to the research papers and download them using 'download_to_pdf'.
    3. give the file name the same as the first 5 characters title of the paper""",
    description='An assistant that can search the web',
    tools=[search_arxiv_tool, download_to_pdf]
)

async def main_async():
    session_id = "session-1"
    user_id = "user_1"
    app_name = "search_appv1"
    usr_input = input(str("What domain are you looking to perform a research in? "))
    runner = InMemoryRunner(agent=search_agent, app_name = app_name)
    user_msg = Content(parts=[Part(text= usr_input)])
    print("--- Search Agent running ---")
    await runner.session_service.create_session(
        session_id= session_id,
        user_id = user_id,
        app_name = app_name
    )
    print("session created successfuly")

    async for event in runner.run_async(
        session_id = session_id,
        user_id = user_id,
        new_message = user_msg,
    ):
        try:
            print("Agent response:")
            print(event)
        except Exception:
            pass


if __name__ == "__main__":
    import asyncio
    asyncio.run(main_async())
