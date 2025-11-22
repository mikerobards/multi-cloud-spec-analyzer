# Multi-Cloud Spec Analyzer Documentation

This project is an automated agent workflow designed to analyze vague project requirements and convert them into structured Azure DevOps Work Items. It leverages a multi-cloud approach, using Google's Gemini for analysis and Azure's GPT-4 for structured output.

## Files

### `main.py`

This is the core application script. It uses **LangGraph** to define a stateful workflow.

**Key Components:**

1.  **Configuration**:
    *   **Google Vertex AI (Gemini 1.5 Pro)**: Acts as the "Reader" brain. It is chosen for its large context window, making it suitable for ingesting large requirement documents.
    *   **Azure OpenAI (GPT-4)**: Acts as the "Writer" brain. It is used to ensure the final output adheres to specific enterprise formatting standards (simulated here).

2.  **Agent State (`AgentState`)**:
    *   A shared dictionary that passes data between steps.
    *   `spec_text`: The input requirements.
    *   `analysis_gaps`: The intermediate analysis from Gemini.
    *   `ado_tickets`: The final JSON output for Azure DevOps.

3.  **Workflow Nodes**:
    *   `analyze_requirements_node`: Sends the spec to Gemini to identify gaps and ambiguities.
    *   `draft_tickets_node`: Sends the spec and Gemini's analysis to Azure OpenAI to draft structured Work Items.

4.  **Execution**:
    *   The script compiles the graph and runs it with a sample vague requirement about a "Customer Loyalty" database migration.

### `requirements.txt`

This file lists the Python dependencies required to run the application.

*   `python-dotenv`: Loads environment variables (API keys, project IDs) from a `.env` file.
*   `langgraph`: The library used to build the stateful, multi-step agent workflow.
*   `langchain-google-vertexai`: The connector for Google Cloud Vertex AI models (Gemini).
*   `langchain-openai`: The connector for Azure OpenAI models.
*   `langchain-core`: Core utilities and base classes for LangChain/LangGraph.
