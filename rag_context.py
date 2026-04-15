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
