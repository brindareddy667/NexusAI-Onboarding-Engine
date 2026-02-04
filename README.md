NexusAI: Intelligent RAG-Driven Onboarding Engine
NexusAI is a professional-grade AI platform that transforms static corporate documents into interactive, personalized training journeys. Using Retrieval-Augmented Generation (RAG), the system ensures that onboarding is data-driven, secure, and verifiable.

🚩 Problem Statement
Information Overload: New hires struggle to retain information from massive, static PDF manuals.

Passive Verification: Traditional systems rely on "read-only" checkboxes rather than proving actual policy comprehension.

Privacy Risks: Using public AI for internal documents risks leaking corporate secrets to the cloud.

One-Size-Fits-All: Standardized training ignores individual skill gaps, leading to time wastage.

🎯 Project Objectives
1.Personalized Paths: Generate 7-day roadmaps tailored to the hire's resume and role requirements.

2.Data Sovereignty: Use local vector indexing to ensure company SOPs never leave the private environment.

3.Active Validation: Enforce "proof-of-work" by requiring GitHub or file submissions to unlock milestones.

4.Automated Compliance: Generate dynamic assessments derived strictly from the provided context.

⚙️ Methodology (The 5 Phases)
1. Knowledge Ingestion & Vectorization
The Admin uploads PDFs which are chunked and converted into 384-dimensional vectors using Sentence Transformers (all-MiniLM-L6-v2) and stored in ChromaDB.

2. Skill-Gap Analysis
Groq (Llama 3.3) analyzes the hire's resume against the indexed SOPs to generate a tailored JSON roadmap that skips redundant training.

3. RAG-Powered Lesson Synthesis
The system retrieves relevant SOP chunks via Semantic Search and synthesizes them into interactive Socratic lessons with specific Technical Challenges.

4. Verification Gate
FastAPI manages a logic gate that locks the final assessment until all GitHub/File proofs are submitted and recorded in SQLite.

5. Dynamic Certification
The AI generates a 15-question exam based solely on the retrieved document context to verify 100% mastery.

🛠️ Tech Stack
1.Inference: Groq (Llama 3.3-70B) for sub-second response times.

2.Vector Store: ChromaDB (Local Persistence).

3.Backend: FastAPI (Python).

4.Database: SQLite3.

5.Embeddings: HuggingFace all-MiniLM-L6-v2.

🚀 Setup Instructions
Clone the repo: git clone https://github.com/brindareddy667/NexusAI-Onboarding-Engine.git

Install Dependencies: pip install -r requirements.txt

Environment: Create a .env file and add GROQ_API_KEY=your_key_here.

Run: python main.py
