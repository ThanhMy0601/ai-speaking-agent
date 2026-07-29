"""Topic context provider.

Note: despite the class name, there is no retrieval-augmented generation
here — no embeddings, no vector search. It is one primary-key SQL lookup
plus a hardcoded guide keyed on the topic *title*, which means renaming a
topic in Rails silently degrades that topic to a generic fallback.

Phase 3 replaces this file with context_client.py, which fetches rendered
topic content over HTTP from Rails and removes both the title-string
coupling and the agent's database credentials.
"""

import os
import psycopg2
import psycopg2.extras
from typing import Optional


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/ai_english_speaking_dev")

GENERIC_CONTEXT = "General English conversation practice."


class RAGContextProvider:
    """Reads topic rows from PostgreSQL for LLM prompt enrichment."""

    def __init__(self):
        self.conn = psycopg2.connect(DATABASE_URL)
        self.conn.autocommit = True

    def get_topic_context(self, topic_id: Optional[int]) -> str:
        """Retrieve topic information and generate practice context."""
        if not topic_id:
            return GENERIC_CONTEXT

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT title, description FROM topics WHERE id = %s AND active = true",
                (topic_id,),
            )
            topic = cur.fetchone()
            if not topic:
                return GENERIC_CONTEXT

        topic_guides = {
            "Everyday Conversations": (
                "Focus on natural daily-life English: greetings, small talk about weather, weekend plans, "
                "hobbies, neighborhood, and current events. Use casual but correct English. "
                "Ask the learner about their day, family, or interests to spark conversation."
            ),
            "Work & Career": (
                "Focus on professional English: job responsibilities, workplace relationships, career goals, "
                "industry trends, and professional skills. Ask about their current job, challenges at work, "
                "or career aspirations. Use business vocabulary naturally."
            ),
            "Travel & Tourism": (
                "Focus on travel English: describing destinations, booking travel, navigating transport, "
                "hotel check-ins, asking for directions, and cultural experiences. "
                "Ask about dream destinations, past travels, or travel tips."
            ),
            "Shopping & Money": (
                "Focus on consumer English: shopping for different items, comparing prices, discussing deals, "
                "banking transactions, budgeting, and financial decisions. "
                "Ask about shopping habits, recent purchases, or money management."
            ),
            "Health & Wellness": (
                "Focus on health English: describing symptoms, visiting doctors, healthy habits, fitness routines, "
                "mental wellness, and nutrition. Ask about their wellness routine, health goals, or medical experiences."
            ),
            "Food & Dining": (
                "Focus on food English: describing tastes and dishes, ordering at restaurants, cooking methods, "
                "dietary preferences, and food culture around the world. "
                "Ask about favorite cuisines, cooking experiences, or restaurant visits."
            ),
            "Technology & Internet": (
                "Focus on tech English: discussing apps, social media, gadgets, cybersecurity, AI, "
                "and the impact of technology on daily life. Ask about their favorite tech, social media use, "
                "or opinions on technology trends."
            ),
            "Education & Learning": (
                "Focus on academic English: discussing study methods, school experiences, learning goals, "
                "academic challenges, and the value of education. Ask about their educational background, "
                "favorite subjects, or learning strategies."
            ),
            "Environment & Nature": (
                "Focus on environmental English: climate change, sustainability, recycling, wildlife, "
                "natural disasters, and eco-friendly living. Ask about environmental concerns, "
                "green habits, or favorite nature experiences."
            ),
            "Culture & Entertainment": (
                "Focus on cultural English: discussing movies, music, sports, books, festivals, "
                "local traditions, and entertainment trends. Ask about favorite films, music genres, "
                "sports they follow, or cultural events they enjoy."
            ),
        }

        guide = topic_guides.get(topic["title"], f"Practice English conversation about {topic['title']}.")
        return f"Topic: {topic['title']}\nDescription: {topic['description']}\n\nConversation Guide:\n{guide}"

    def close(self):
        if self.conn:
            self.conn.close()
