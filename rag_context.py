"""RAG Context Provider — retrieves lesson-specific content from pgvector."""

import os
import psycopg2
import psycopg2.extras
from typing import Optional


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/ai_english_speaking_dev")


class RAGContextProvider:
    """Connects to PostgreSQL/pgvector to retrieve lesson-specific vocabulary,
    grammar rules, and conversation context for LLM prompt enrichment."""

    def __init__(self):
        self.conn = psycopg2.connect(DATABASE_URL)
        self.conn.autocommit = True

    def get_lesson_context(self, lesson_id: Optional[int]) -> str:
        """Retrieve vocabulary and grammar context for a specific lesson."""
        if not lesson_id:
            return "General English conversation practice."

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get lesson info
            cur.execute(
                "SELECT title, description, vocabulary_data, grammar_data, dialogue_data "
                "FROM lessons WHERE id = %s",
                (lesson_id,),
            )
            lesson = cur.fetchone()
            if not lesson:
                return "General English conversation practice."

            # Get vocabulary items
            cur.execute(
                "SELECT word, phonetic, definition, example_sentence "
                "FROM vocabulary_items WHERE lesson_id = %s",
                (lesson_id,),
            )
            vocab_items = cur.fetchall()

            context_parts = [
                f"Current Lesson: {lesson['title']}",
                f"Description: {lesson['description']}",
            ]

            if vocab_items:
                vocab_text = "\n".join(
                    f"- {v['word']} ({v['phonetic']}): {v['definition']}"
                    for v in vocab_items
                )
                context_parts.append(f"Key Vocabulary:\n{vocab_text}")

            if lesson.get("grammar_data"):
                grammar = lesson["grammar_data"]
                if isinstance(grammar, dict) and "topics" in grammar:
                    context_parts.append(
                        f"Grammar Focus: {', '.join(grammar['topics'])}"
                    )

            return "\n\n".join(context_parts)

    def get_topic_context(self, topic_id: Optional[int]) -> str:
        """Retrieve topic information and generate practice context."""
        if not topic_id:
            return "General English conversation practice."

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT title, description FROM topics WHERE id = %s AND active = true",
                (topic_id,),
            )
            topic = cur.fetchone()
            if not topic:
                return "General English conversation practice."

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

    def get_ielts_context(self, part: int) -> str:
        """Retrieve IELTS-specific context for a given test part."""
        contexts = {
            1: (
                "IELTS Speaking Part 1 - Introduction and Interview (4-5 minutes).\n"
                "Ask 4-5 questions about familiar topics: home, family, work, studies, interests.\n"
                "Questions should be straightforward. Assess fluency and basic communication."
            ),
            2: (
                "IELTS Speaking Part 2 - Long Turn (3-4 minutes including 1 min prep).\n"
                "Present a topic card with bullet points. Give 1 minute preparation time.\n"
                "The candidate should speak for 1-2 minutes. Ask 1-2 follow-up questions."
            ),
            3: (
                "IELTS Speaking Part 3 - Discussion (4-5 minutes).\n"
                "Ask 4-6 abstract discussion questions related to the Part 2 topic.\n"
                "Probe deeper with follow-ups. Assess ability to express and justify opinions."
            ),
        }
        return contexts.get(part, contexts[1])

    def get_roleplay_context(self, scenario: str) -> str:
        """Retrieve role-play scenario context."""
        scenarios = {
            "salary_negotiation": (
                "You are a hiring manager at a tech company. The candidate has received an offer "
                "and wants to negotiate salary. Be professional but firm. The initial offer is "
                "$85,000. You have flexibility up to $95,000. Discuss benefits, equity, and growth."
            ),
            "client_presentation": (
                "You are a potential client evaluating a software solution. Ask probing questions "
                "about features, pricing, implementation timeline, and support. Be skeptical but fair."
            ),
            "job_interview": (
                "You are a senior hiring manager conducting a behavioral interview. Ask about "
                "past experiences, challenges overcome, teamwork, and leadership. Use the STAR method."
            ),
            "team_meeting_facilitation": (
                "You are a team member in a project meeting. The learner is facilitating. "
                "Occasionally go off-topic, disagree constructively, and ask clarifying questions."
            ),
            "conflict_resolution": (
                "You are a colleague who disagrees with the learner's project approach. "
                "Express concerns professionally. Be open to compromise but defend your position."
            ),
        }
        return scenarios.get(scenario, scenarios["job_interview"])

    def close(self):
        if self.conn:
            self.conn.close()
