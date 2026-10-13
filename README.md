

\# Bank RAG MCP



Application d'analyse de rapports annuels bancaires utilisant

Retrieval-Augmented Generation (RAG), une base vectorielle et MCP.



\## Objectifs

\- Extraire le texte des rapports annuels bancaires PDF.

\- Découper les documents en chunks avec leurs références de page.

\- Générer des embeddings et effectuer une recherche sémantique.

\- Produire des réponses fondées sur les documents et citer les sources.

\- Exposer les fonctions de recherche avec un serveur MCP.

\- Évaluer la pertinence des réponses.



\## Technologies

\- Python 3.10

\- PyMuPDF

\- LangChain

\- Sentence Transformers

\- ChromaDB

\- Model Context Protocol (MCP)



\## Structure

\- `data/raw/` : PDF locaux.

\- `data/processed/` : données extraites.

\- `src/ingestion/` : extraction et découpage.

\- `src/retrieval/` : recherche vectorielle.

\- `src/generation/` : génération des réponses.

\- `src/mcp\_server/` : serveur MCP.

\- `tests/` : tests automatisés.

\- `evaluation/` : évaluation du système.



