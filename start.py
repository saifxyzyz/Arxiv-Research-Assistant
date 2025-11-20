from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
import arxiv
import os
import requests
from pypdf import PdfReader
from google.adk.agents import SequentialAgent
from fpdf import FPDF


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

def get_all_papers_content() -> str:
    if not os.path.exists(DOWNLOAD_DIR):
        return "No papers found."
    combined_text = "Here is the content of the research papers I found:\n\n"
    files = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".pdf")]
    if not files:
        return "No PDF files found in the directory."

    print(f"[System] Reading {len(files)} PDF files from disk...")

    for filename in files:
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        try:
            reader = PdfReader(filepath)
            paper_text = ""
            # Read all pages
            for page in reader.pages:
                paper_text += page.extract_text()
            # Add to the giant string with clear separators
            combined_text += f"=== START OF PAPER: {filename} ===\n"
            combined_text += paper_text[:5000] # Limit to 50k chars per paper to be safe, or remove limit for full text
            combined_text += f"\n=== END OF PAPER: {filename} ===\n\n"
            print(f"   -> Loaded: {filename}")
        except Exception as e:
            print(f"   -> Failed to read {filename}: {e}")

    combined_text += "\nINSTRUCTIONS: Write a detailed report. Create a dedicated section for EACH paper listed above."
    return combined_text

def write_to_pdf(filename,text):
    print(f"[Report Writer] writing report to: {filename}")
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        safe_content = text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, safe_content)
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        pdf.output(filename)
        return f"Successfully saved report to {filename}"

    except Exception as e:
        return f"Error creating PDF: {str(e)}"


search_agent = Agent(
    name="search_assistant",
    model = "gemini-2.0-flash",
    instruction="""You are a research assistant.
    1. search for the latest research papers.
    2. Find the direct pdf files to the research papers and download them using 'download_to_pdf'.
    3. give the file name the same as the first 5 characters title of the paper, replace any " " with "_" """,
    description='An assistant that can search the web',
    tools=[search_arxiv_tool, download_to_pdf]
)

report_agent = Agent(
    name="search_assistant",
    model = "gemini-2.0-flash",
    instruction="""You are a senior technical writer. You read research papers and summarize them into structured reports.
    1. To access the contents of the research papers use the 'get_all_papers_content' tool
    2. I want you to create a detailed report about all the research papers with different sections for each research paper""",
    tools=[get_all_papers_content]
)

root_agent = SequentialAgent(
    name="root_agent",
    sub_agents = [search_agent, report_agent],
    description= "Manages the execution of sub agents"
)

async def main_async():
    session_id = "session-1"
    user_id = "user_1"
    app_name = "search_appv1"
    usr_input = input(str("What domain are you looking to perform a research in? "))
    runner = InMemoryRunner(agent=root_agent, app_name = app_name)
    user_msg = Content(parts=[Part(text= usr_input)])
    print("--- Root Agent running ---")
    await runner.session_service.create_session(
        session_id= session_id,
        user_id = user_id,
        app_name = app_name
    )
    print("session created successfuly")
    full_report_text = ""
    async for event in runner.run_async(
        session_id = session_id,
        user_id = user_id,
        new_message = user_msg,
    ):
        if hasattr(event, 'content') and event.content:
            if hasattr(event.content, 'parts'):
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        # Print to console (streaming)
                        print(part.text, end="", flush=True)
                        # Capture for PDF
                        full_report_text += part.text
        if full_report_text:
            write_to_pdf("Final_Research_Report.pdf", full_report_text)
        else:
            print("[System] No text generated, skipping PDF creation.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main_async())
