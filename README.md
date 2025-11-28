# Project Chimera: An AI Research Synthesis Platform

**A submission for the Google ADK Kaggle Hackathon.**

Project Chimera is a sophisticated, AI-powered research automation platform designed to accelerate innovation. It leverages a multi-agent system built with the Google Agent Development Kit (ADK) to intelligently source, analyze, and synthesize the latest scientific literature from arXiv. By transforming raw research papers into structured, easy-to-digest reports, Chimera empowers researchers, engineers, and enthusiasts to stay at the cutting edge of their fields with unparalleled efficiency.

![Dataflow Diagram](https://github.com/user-attachments/assets/8d5f035a-f76f-4c78-ab48-45061e9e7310)
*Figure 1: The Multi-Agent Architecture of Project Chimera, showcasing the seamless flow of data from ingestion to final report generation.*

## Advanced Features

- **Autonomous Multi-Agent Collaboration:** At its core, Chimera utilizes a sophisticated `SequentialAgent` to orchestrate a team of specialized AI agents. This ensures a robust, end-to-end workflow from discovery to delivery.
- **Dynamic Data Ingestion Pipeline:** The system intelligently queries the arXiv API to fetch the most relevant and recent research papers based on user-defined domains. It includes a resilient download mechanism to handle various paper formats and ensure data integrity.
- **LLM-Powered Analysis & Synthesis:** Chimera harnesses the power of Google's Gemini 2.0 Flash model to perform deep analysis of the ingested research papers. The `report_agent` reads the full text of multiple papers and generates a comprehensive, structured report, complete with dedicated sections for each paper.
- **On-the-Fly PDF Report Generation:** The final output is a professionally formatted PDF document, synthesized in real-time. This allows for easy sharing, archiving, and offline access to the synthesized research.
- **Scalable and Modular Architecture:** Built on the Google ADK, the entire system is designed to be scalable and modular. New agents, data sources, or processing capabilities can be seamlessly integrated to expand the platform's functionality.

## Architectural Overview

Project Chimera is built on a robust, multi-agent architecture, as illustrated in the dataflow diagram above. The workflow is orchestrated by a `root_agent` (`SequentialAgent`) that manages the execution of the following specialized agents:

1.  **Search Agent (`search_assistant`):** This agent is responsible for the initial discovery phase. It takes a user's query, interfaces with the `search_arxiv_tool`, and identifies the most relevant research papers. It then uses the `download_to_pdf` tool to ingest the papers into the local filesystem.
2.  **Report Agent (`report_agent`):** Once the papers are downloaded, this agent takes over. It uses the `get_all_papers_content` tool to read and process the full text of the papers. Leveraging the advanced reasoning capabilities of Gemini 2.0, it synthesizes the information into a single, coherent report.

This decoupled, agent-based architecture ensures that each part of the process is handled by a specialized component, leading to a more robust and maintainable system.

## Technology Stack

- **Core Framework:** Google Agent Development Kit (ADK)
- **Language Model:** Google Gemini 2.0 Flash
- **Programming Language:** Python
- **Key Libraries:**
    - `google-adk`
    - `arxiv`
    - `pypdf`
    - `fpdf`
    - `requests`

## Installation and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/saifxyzyz/Arxiv-Research-Assistant.git
    cd Arxiv-Research-Assistant
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Google API Key:**
    Make sure you have a `.env` file with your Google API key:
    ```
    GOOGLE_API_KEY=your_api_key_here
    ```

5.  **Run the application:**
    ```bash
    python start.py
    ```
    You will be prompted to enter a research domain. The final report will be saved in the `papers/<your-domain>/` directory.

## Future Work

Project Chimera is a foundational platform with immense potential for growth. Future enhancements could include:

- **Integration with more data sources:** Expanding beyond arXiv to include sources like PubMed, IEEE Xplore, and others.
- **Interactive Web Interface:** Building a front-end to provide a more user-friendly experience for interacting with the platform.
- **Knowledge Graph Integration:** Creating a knowledge graph from the synthesized research to uncover hidden connections and insights.
- **Advanced Querying:** Allowing for more complex, natural language queries to drive the research process.
