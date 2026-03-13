from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import BASE_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL

# ==============================================================================
# EXHAUSTIVE KNOWLEDGE REGISTRY (THE "PERFECT" ANSWER ENGINE)
# ==============================================================================
KNOWLEDGE_REGISTRY = {
    "project_overview": {
        "keywords": ["urbansightai", "project", "proposal", "what is", "mission", "vision"],
        "content": "UrbanSightAI is a policy-first platform for Bangladesh that bridges the gap between ward-level evidence and municipal finance. It converts spatial data into budget-ready outputs like Top-N worklists and strategic memos. It uses AI to optimize urban micro-works for impact and equity."
    },
    "dustbin_waste": {
        "keywords": ["dustbin", "waste", "garbage", "trash", "ময়লা", "ডাস্টবিন", "জমা", "পরিষ্কার"],
        "content": "UrbanSightAI manages waste and dustbin issues through: 1. Citizen reports via this Chatbot (Blocked Drains, Waste Hotspots). 2. A segmentation model that identifies informal waste patches from imagery. 3. The Policy Ranker, which prioritizes setting up new waste collection points and clearing hotspots based on impact (population reached) and cost."
    },
    "drainage_waterlogging": {
        "keywords": ["drain", "drainage", "waterlogging", "flood", "infrastructure"],
        "content": "To solve drainage and waterlogging, UrbanSightAI: 1. Uses U-Net segmentation to count and map blocked drains. 2. Periodic drain cleaning is ranked as a 'Top-N' intervention. 3. Scenario simulations help planners forecast budget needs to reduce households at risk during floods."
    },
    "road_maintenance": {
        "keywords": ["road", "street", "repair", "maintenance", "রাস্তা", "সড়ক", "মেরামত"],
        "content": "UrbanSightAI identifies road service gaps using a Digital Twin Dashboard. The system generates procurement-ready briefs for road repairs, scoring them by cost-effectiveness and equity to ensure underserved areas are prioritized."
    },
    "budget_personnel": {
        "keywords": ["budget", "personnel", "cost", "salary", "বাজেট", "বেতন", "খরচ"],
        "content": "Personnel Budget (10.80 lakh Taka): - AI/ML Engineers: 3.6 lakh (2 units). - Geospatial Analyst: 1.65 lakh. - Backend Developer: 1.50 lakh. - Frontend Developer: 1.35 lakh. - Policy Researcher: 1.20 lakh. - Field Enumerators: 1.5 lakh."
    },
    "budget_operations": {
        "keywords": ["operational", "cloud", "data", "hosting", "workshop", "অপারেশনাল", "হোস্টিং"],
        "content": "Operational Budget (2.85 lakh Taka): - Cloud hosting (VM/GPU): 0.60 lakh. - Data acquisition: 0.75 lakh. - Software licenses: 0.40 lakh. - Communication/Outreach: 0.50 lakh. - Training workshops: 0.60 lakh."
    },
    "budget_total": {
        "keywords": ["total cost", "total budget", "মোট বাজেট", "মোট খরচ"],
        "content": "The TOTAL PROJECT COST of UrbanSightAI is 19.71 lakh Taka. This includes Personnel (10.80L), Operational (2.85L), Capital (3.50L), and Administrative Overhead (2.56L)."
    },
    "sdg_11_3": {
        "keywords": ["11.3", "planning", "sustainable planning", "পরিকল্পনা"],
        "content": "Target 11.3: Improve inclusive and sustainable urban planning. UrbanSightAI addresses this by providing data-driven tools for ward-level planners to make fair, evidence-based development decisions."
    },
    "sdg_11_5": {
        "keywords": ["11.5", "disaster", "risk", "দুর্যোগ"],
        "content": "Target 11.5: Reduce disaster impact through early signals and resource planning. The platform uses flood exposure maps and population-at-risk indicators to prioritize resilience projects."
    },
    "sdg_11_6": {
        "keywords": ["11.6", "environment", "waste management", "পরিবেশ"],
        "content": "Target 11.6: Reduce environmental burden via better waste and drain prioritization. UrbanSightAI maps waste hotspots and ensures cleaning work-orders are issued to the most critical areas."
    },
    "ai_ranker": {
        "keywords": ["ranker", "lightgbm", "xgboost", "prioritization", "র‍্যাঙ্কার"],
        "content": "The AI Prioritization Ranker (LightGBM/XGBoost) scores candidate micro-works by impact-per-cost, feasibility, and equity weighting. It outputs a Top-N list of projects that provide the most benefit to citizens per BDT spent."
    },
    "ai_segmentation": {
        "keywords": ["segmentation", "u-net", "unet", "evidence extractor", "সেগমেন্টেশন"],
        "content": "The Evidence Extractor (U-Net/DeepLab) converts satellite and field imagery into policy indicators, such as mapping informal settlements, service gaps, and blocked-drain counts for the digital twin."
    },
    "deliverables": {
        "keywords": ["deliverable", "deliverables", "outcome", "outputs"],
        "content": "UrbanSightAI Core Deliverables: 1. AI Policy Prioritization (Top-N lists). 2. Global AI Expert Assistant (RAG interface). 3. Ward Digital Twin Dashboard. 4. Strategic Scenario Engine (Memos for finance officers). 5. Unified Planner Console."
    },
    "digital_twin": {
        "keywords": ["digital twin", "dashboard", "map", "ডিজিটাল টুইন", "ড্যাশবোর্ড", "ম্যাপ"],
        "content": "The Digital Twin Dashboard is an interactive map that summarizes ward assets, service gaps (exposed population, blocked drains), and exposure indicators. It helps planners identify where interventions are needed visually and instantly."
    },
    "revenue_sustainability": {
        "keywords": ["revenue", "revenue model", "sustainability", "money", "আয়", "আয়ের পথ"],
        "content": "UrbanSightAI earns revenue through: 1. Municipal Subscriptions (B2G). 2. Data-as-a-Service (APIs for researchers). 3. Consultancy & Integration. 4. Performance-based results contracts."
    }
}

# ==============================================================================
# QUERY PREPROCESSOR
# ==============================================================================
class QueryPreprocessor:
    @staticmethod
    def clean(query: str) -> str:
        slogans = [
            r"joy\s?bangla", r"জয়\s?বাংলা", r"joy\s?mujib", r"জয়\s?মুজিব",
            r"hi", r"hello", r"hey", r"please", r"kindly", r"তদন্ত\s?করুন"
        ]
        text = query.lower()
        for s in slogans:
            text = re.sub(s, "", text)
        return text.strip()

# ==============================================================================
# RAG SERVICE ENGINE (LLAMA2 COMPATIBLE)
# ==============================================================================
class RAGService:
    def __init__(self, knowledge_path: Path = BASE_DIR.parent / "proposal.txt") -> None:
        self.knowledge_path = knowledge_path
        self.chunks: list[str] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self.preprocessor = QueryPreprocessor()

    def load_knowledge(self) -> None:
        target = self.knowledge_path
        if not target.exists():
            fallback = Path("D:/UrbanSightAI-main/proposal.txt")
            target = fallback if fallback.exists() else None
        
        if not target:
            print("CRITICAL Error: Knowledge proposal.txt is missing.")
            return

        with target.open("r", encoding="utf-8") as f:
            content = f.read()

        raw_chunks = content.split("\x0c")
        processed_chunks = []
        for rc in raw_chunks:
            sections = rc.split("\n\n")
            for section in sections:
                clean = section.strip()
                if len(clean) > 80:
                    processed_chunks.append(clean)
        
        self.chunks = processed_chunks
        if self.chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
            print(f"Ultimate RAG (Llama2 Mode): Loaded {len(self.chunks)} items.")

    def hybrid_retrieve(self, query: str, top_k: int = 4) -> list[str]:
        cleaned_query = self.preprocessor.clean(query)
        retrieved_sources = []
        
        # --- LAYER 1: PRIORITY KEYWORD REGISTRY ---
        # Checking ALL registry segments to ensure "Perfect" matching
        for key, entry in KNOWLEDGE_REGISTRY.items():
            if any(kw in cleaned_query for kw in entry["keywords"]):
                retrieved_sources.append(f"[Verified Fact]: {entry['content']}")
        
        # --- LAYER 2: TF-IDF SEMANTIC SEARCH ---
        if self.chunks and self.tfidf_matrix is not None:
            query_vec = self.vectorizer.transform([cleaned_query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            for i in top_indices:
                if similarities[i] > 0.08: # Slightly lower to catch more
                    retrieved_sources.append(f"[Proposal Context]: {self.chunks[i]}")

        # Unique results only
        seen = set()
        return [x for x in retrieved_sources if not (x in seen or seen.add(x))]

    def classify_intent(self, query: str) -> str:
        cleaned = self.preprocessor.clean(query)
        
        # Scoring logic
        scores = {
            "CITIZEN_COMPLAINT": 0,
            "PLANNER_ADVICE": 0,
            "DATA_QUERY": 0
        }
        
        complaint_kws = ["blocked", "trash", "garbage", "drain", "stink", "flood", "waterlogging", "ময়লা", "জমা", "পরিষ্কার"]
        advice_kws = ["budget", "allocation", "recommend", "advice", "strategy", "priority", "spending", "lakh", "পরামর্শ", "বাজেট"]
        data_kws = ["what is the", "tell me about", "stats", "indicators", "score", "how many", "population", "risk"]
        
        for kw in complaint_kws:
            if kw in cleaned: scores["CITIZEN_COMPLAINT"] += 2
        for kw in advice_kws:
            if kw in cleaned: scores["PLANNER_ADVICE"] += 2
        for kw in data_kws:
            if kw in cleaned: scores["DATA_QUERY"] += 1
            
        # Tie-breaker or default
        max_score = max(scores.values())
        if max_score == 0:
            return "GENERAL"
            
        # Return the highest score intent
        return max(scores, key=scores.get)

    async def chat(self, query: str, ward_context: str | None = None, chat_mode: str | None = None) -> dict[str, Any]:
        intent = chat_mode if chat_mode else self.classify_intent(query)
        sources = self.hybrid_retrieve(query)
        
        if not sources and not ward_context:
            return {
                "answer": "Data not found in UrbanSightAI proposal. I can only provide insights based on the project's evidence and policy facts.", 
                "sources": []
            }
            
        context = "\n---\n".join(sources)
        if ward_context:
            context = f"LIVE WARD DATA CONTEXT:\n{ward_context}\n\n---\nPROPOSAL CONTEXT:\n{context}"
        
        # Tailor identity based on intent
        role_desc = "UrbanSightAI Assistant"
        rule_set = "ONLY answer based on the FACTS and LIVE DATA above."
        
        if intent == "CITIZEN_COMPLAINT":
            role_desc = "UrbanSightAI Citizen Support Liaison"
            rule_set += " Acknowledge the citizen's concern. MOST IMPORTANTLY, based on the context provided, propose a SPECIFIC SOLUTION or actionable next step the city should take to fix the reported issue."
        elif intent == "PLANNER_ADVICE":
            role_desc = "UrbanSightAI Policy Consultant"
            rule_set += " Provide professional budget and strategy recommendations based on the data."
        elif intent == "DATA_QUERY":
            role_desc = "UrbanSightAI Data Analyst"
            rule_set += " Provide precise numbers and indicators from the ward context."

        global_context = f"Identity: You are the {role_desc} for Bangladesh. You ONLY answer in English."
        
        full_prompt = f"""<|system|>
{global_context}

CONTEXT:
{context}

RULES:
1. {rule_set}
2. If the answer is not in the context, say: "Data not found in UrbanSightAI proposal or current ward data."
3. ALWAYS respond in English.</s>
<|user|>
{query}</s>
<|assistant|>
"""

        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1, # Slightly higher for more natural roleplay
                "num_predict": 400,
                "stop": ["</s>", "<|user|>", "<|system|>", "Question:"]
            }
        }

        try:
            print(f"Chat Request ({OLLAMA_MODEL}) | Intent: {intent}...")
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            answer = response.json().get("response", "").strip()
            
            # Clean possible repetition or prefixes
            answer = re.sub(r"^(answer|assistant|response):\s?", "", answer, flags=re.I)
            return {"answer": answer, "sources": sources, "intent": intent}
        except Exception as e:
            return {"answer": f"System Busy: {str(e)}", "sources": sources}

    async def consult_scenario(self, vision: str) -> dict[str, Any]:
        """Translates a planning vision into specific simulation weights using LLM."""
        prompt = f"""<|system|>
You are a Senior Urban Planner for Bangladesh. Your task is to translate a user's strategic "Vision" into technical simulation weights.
The weights will control an AI Policy Ranker.

AVAILABLE WEIGHTS (0.0 to 0.4 each):
- "impact_per_lakh": Focus on cost-efficiency and maximum ROI.
- "equity_need": Focus on low-income, informal, and underserved populations.
- "urgency": Focus on immediate safety, flood response, and critical crises.
- "feasibility": Focus on projects with high technical success probability.
- "beneficiary_norm": Focus on reaching the highest absolute number of residents.
- "prior_rank_norm": Stability; bias towards existing high-ranked evidence.
- "readiness_norm": Bias towards "shovel-ready" projects with minimal permits needed.

AVAILABLE SECTORS:
- "Drainage": Flood safety, canals, drain blockage.
- "Water": Clean water access, informal settlement supply.
- "Waste": Solid waste management, cleanliness.
- "Road": Connectivity, repair, emergency access.
- "Green": Tree cover, parks, urban heat reduction.
- "Public Safety": Street lighting, community safety.

RULES:
1. Return ONLY a raw JSON object. 
2. NO markdown backticks. NO introductory text. NO commentary.
3. "weights" must sum to 1.0 (controlling project selection).
4. "sector_priorities" must be 0.0 to 1.0 for EACH of the 6 sectors (controlling budget distribution).
5. "sector_rationales" must provide a brief, vision-aligned reason for each sector's priority (max 100 chars each).
6. Include a global "reasoning" field.

Example Input: "Help poor people during floods now!"
Example Output:
{{
  "weights": {{"impact_per_lakh": 0.1, "equity_need": 0.35, "urgency": 0.3, "feasibility": 0.05, "beneficiary_norm": 0.1, "prior_rank_norm": 0.05, "readiness_norm": 0.05}},
  "sector_priorities": {{"Drainage": 0.9, "Water": 0.5, "Waste": 0.4, "Road": 0.2, "Green": 0.3, "Public Safety": 0.2}},
  "sector_rationales": {{
    "Drainage": "Directly tackles flood risks for vulnerable households as requested.",
    "Water": "Essential for informal areas prone to flooding and water shortages.",
    "Waste": "Reduces drain blockage which is critical for monsoon readiness.",
    "Road": "Minimal priority unless needed for emergency rescue access.",
    "Green": "Secondary focus; prioritised for local climate cooling.",
    "Public Safety": "Maintained at baseline unless safety issues arise."
  }},
  "reasoning": "Weighted heavily for equity and urgency to tackle flood response in high-need wards."
}}
</s>
<|user|>
Vision: "{vision}"
Respond with JSON ONLY.
</s>
<|assistant|>
{{
"""
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 300,
                "stop": ["</s>", "<|user|>", "<|system|>"]
            }
        }

        try:
            print(f"Consultation Request ({OLLAMA_MODEL}) | Vision: {vision[:40]}...")
            response = requests.post(url, json=payload, timeout=40)
            response.raise_for_status()
            raw_text = response.json().get("response", "").strip()
            # 1. Clear common markdown and conversational fluff
            raw_text = re.sub(r'```(?:json)?', '', raw_text).strip()
            raw_text = re.sub(r'^.*?(\{)', r'\1', raw_text, flags=re.DOTALL) # Strip leading text
            
            # 2. Try to find the first complete JSON object
            match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
            if match:
                try:
                    # Pre-fix the leading '{' we might have added in prompt if needed
                    text_to_parse = match.group(1)
                    if not text_to_parse.startswith("{"):
                        text_to_parse = "{" + text_to_parse
                    return json.loads(text_to_parse)
                except json.JSONDecodeError:
                    pass
            
            # 3. Fallback: return balanced weights instead of crashing
            balanced_weights = {
                "impact_per_lakh": 0.2, "equity_need": 0.2, "urgency": 0.2,
                "feasibility": 0.1, "beneficiary_norm": 0.1,
                "prior_rank_norm": 0.1, "readiness_norm": 0.1
            }
            default_priorities = {s: 0.5 for s in ["Drainage", "Water", "Waste", "Road", "Green", "Public Safety"]}
            
            print(f"LLM Response not parseable. Content: {raw_text[:100]}")
            return {
                "weights": balanced_weights,
                "sector_priorities": default_priorities,
                "sector_rationales": {s: "Balanced priority based on ward baseline indicators." for s in default_priorities},
                "reasoning": "Fallback to balanced weights due to parsing error."
            }
        except Exception as e:
            raw_info = locals().get("raw_text", "No response")
            print(f"Consultation Parse Error: {e} | Raw: {raw_info[:100]}")
            default_priorities = {s: 0.5 for s in ["Drainage", "Water", "Waste", "Road", "Green", "Public Safety"]}
            return {
                "weights": {
                    "impact_per_lakh": 0.2, "equity_need": 0.2, "urgency": 0.2,
                    "feasibility": 0.1, "beneficiary_norm": 0.1,
                    "prior_rank_norm": 0.1, "readiness_norm": 0.1
                },
                "sector_priorities": default_priorities,
                "sector_rationales": {s: "Balanced priority due to system timeout or error." for s in default_priorities},
                "reasoning": f"Fallback to balanced weights due to parsing error: {str(e)}"
            }

RAG_SERVICE = RAGService()
RAG_SERVICE.load_knowledge()
