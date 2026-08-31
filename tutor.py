"""Socratic tutor with phased lesson flow."""

import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from models import get_model
from topics import get_topic

RETURN_PATTERN = re.compile(
    r"\b(i'?m back|i am back|i'?m here|i am here|done reading|finished reading|"
    r"read it|read them|back now|ready to talk|let'?s talk|let'?s go|let us go|"
    r"i read it|i'?ve read|i have read|finished the (article|reading|link)|"
    r"ready to (start|continue)|ok(?:ay)?(?: i'?m)? ready|i skimmed|"
    r"went through it|done with the reading|i went through|read through)\b",
    re.IGNORECASE,
)
CONFIDENT_PATTERN = re.compile(
    r"\b(i'?m confident|i understand|got it|no doubts|no questions|i'?m ready|"
    r"let'?s (build|start|practice)|sounds good)\b",
    re.IGNORECASE,
)
FORGOT_PATTERN = re.compile(
    r"\b(forgot|don'?t remember|what is an? |what'?s an? |wait,? what|"
    r"can you explain again|i'?m lost|confused again)\b",
    re.IGNORECASE,
)
DONE_PATTERN = re.compile(
    r"\b(no doubts|all good|that'?s it|we'?re done|i'?m done|looks good|"
    r"no more questions)\b",
    re.IGNORECASE,
)
FINISH_PATTERN = re.compile(
    r"\b(no doubts|no more questions|that'?s all|i'?m good|all good|"
    r"nothing else|thanks(?: you)?|thank you|end session|all set|bye)\b",
    re.IGNORECASE,
)
IDK_PATTERN = re.compile(
    r"\b(i don'?t know|i dunno|no idea|not sure|idk|i have no idea)\b",
    re.IGNORECASE,
)

FRESHMAN_RULES = """
## Audience: freshman / sophomore undergrad
- Assume this may be their FIRST SE course. No job experience, no cert-prep tone.
- Use plain language and everyday analogies (library, pizza delivery, dorm Wi‑Fi).
- Prefer generic terms: "cloud storage", "CDN", "web server" — NOT AWS/Azure/GCP product names
  during **checking** and **practice** unless the student said them first.
- If the student is stuck ("I don't know"): give ONE hint or analogy only — never the full answer.
  After 2+ "I don't know" on the same idea, give a small nudge (category or first letter) — still
  not the complete solution. Save full names/answers for **complete** phase only.
- During **checking**: stay conceptual. One question per message. Use the suggested check question.
- During **practice**: one step per message. Student proposes ideas; you coach.
- Do NOT turn the session into a vendor quiz (e.g. "name the AWS service") unless the topic is
  explicitly about that vendor.
"""

MAX_COMPLETE_FOLLOWUPS = 3

PHASE_LABELS = {
    "reading": "Active — read the link, then say you're back",
    "checking": "Active — quick check questions",
    "practice": "Active — hands-on practice",
    "complete": "Complete — any last doubts?",
    "ended": "Ended — transcript only",
}


def session_status(session: "Session") -> str:
    if session.ended:
        return "ended"
    if session.phase == "complete":
        return "complete"
    return "active"


@dataclass
class Session:
    session_id: str
    topic_id: str
    provider: str
    personality: str = "confused"
    phase: str = "reading"
    messages: list[dict] = field(default_factory=list)
    questions_asked: int = 0
    practice_steps: int = 0
    practice_stage: int = 0
    awaiting_answer: bool = False
    ended: bool = False
    complete_followups: int = 0

    def to_public(self) -> dict:
        topic = get_topic(self.topic_id)
        return {
            "session_id": self.session_id,
            "topic_id": self.topic_id,
            "topic_name": topic["name"],
            "provider": self.provider,
            "personality": self.personality,
            "phase": self.phase,
            "phase_label": PHASE_LABELS.get(self.phase, self.phase),
            "status": session_status(self),
            "questions_asked": self.questions_asked,
            "practice_steps": self.practice_steps,
            "complete_followups": self.complete_followups,
            "complete_followups_remaining": max(
                0, MAX_COMPLETE_FOLLOWUPS - self.complete_followups
            ),
            "ended": self.ended,
            "messages": self.messages,
        }


def _opening_message(topic_id: str) -> str:
    topic = get_topic(topic_id)
    primary = topic["resource"]
    alt = topic["alt_resource"]
    return (
        f"Today we're working on **{topic['name']}**. I'll coach you through it — "
        f"you do the thinking, I'll guide with questions.\n\n"
        f"First, skim one of these so we share some vocabulary:\n\n"
        f"1. [{primary['title']}]({primary['url']})\n"
        f"2. [{alt['title']}]({alt['url']})\n\n"
        f"No need to memorize everything. When you're ready to continue, just let me know — "
        f"e.g. \"ready\", \"I read it\", or \"let's go\"."
    )


def create_session(session_id: str, topic_id: str, provider: str, personality: str) -> Session:
    session = Session(
        session_id=session_id,
        topic_id=topic_id,
        provider=provider,
        personality=personality,
        phase="reading",
    )
    session.messages.append({"role": "assistant", "content": _opening_message(topic_id)})
    return session


def _history_for_llm(messages: list[dict]) -> list:
    out = []
    for msg in messages:
        if msg["role"] == "assistant":
            out.append(AIMessage(content=msg["content"]))
        else:
            out.append(HumanMessage(content=msg["content"]))
    return out


def _practice_stage_info(session: Session) -> tuple[dict | None, list[dict]]:
    topic = get_topic(session.topic_id)
    stages = topic.get("practice_stages", [])
    if not stages or session.practice_stage >= len(stages):
        return None, stages
    return stages[session.practice_stage], stages


def _build_tutor_system(session: Session) -> str:
    topic = get_topic(session.topic_id)
    questions = topic["check_questions"]
    q_hint = ""
    if session.phase == "checking" and session.questions_asked < len(questions):
        q_hint = f"\nSuggested next check-in theme: {questions[session.questions_asked]}"

    stage, all_stages = (None, [])
    practice_hint = ""
    if session.phase == "practice":
        stage, all_stages = _practice_stage_info(session)
        if stage:
            order = " → ".join(s["label"] for s in all_stages)
            practice_hint = f"""
## Current practice stage: {stage['label']} (step {session.practice_stage + 1}/{len(all_stages)})
Stage focus: {stage['focus']}
Recommended build order: {order}
If the student jumps ahead (e.g. boundary before actors), briefly acknowledge their idea,
then nudge them back to the current stage. Do NOT give the full solution yet.
"""
    else:
        all_stages = get_topic(session.topic_id).get("practice_stages", [])

    return f"""You are a software-engineering tutor for undergrad students.

{FRESHMAN_RULES}

## Tutoring approach: self-directed learning
- The learning goal is fixed: help the student master **{topic['name']}** in this session.
- You are a coach, not a lecturer. The student does the thinking; you facilitate.
- Ask eliciting questions FIRST: "What's your understanding?" / "How would you define that?"
- After they attempt an answer, guide with short hints — don't dump full explanations.
- If they ask "what is X?", give a 1–2 sentence plain-English definition, then ask them to try again.
- Praise reasoning and effort, not only correct answers.
- Keep replies short (2–4 sentences unless defining a term they asked about).

## Topic
{topic['name']}

## Current phase: {session.phase}
{PHASE_LABELS.get(session.phase, "")}

## Phase rules
- **reading**: Student should skim the link first. If they haven't returned yet, encourage
  them to read — do NOT quiz yet. Accept any message that signals they're ready to continue.
- **checking**: Ask ONE check question at a time ({len(questions)} total). Questions must be
  conceptual and freshman-friendly — "in your own words", simple examples. No vendor/product trivia.
  Use the suggested check question for this turn. Do NOT skip to advanced scenarios.
- **practice**: {topic['practice_prompt']}
  ONE guiding question per turn. Generic concepts first. Follow practice stage order below.
  Never give the complete answer until practice is done.
- **complete**: Lesson is DONE. NOW you may share the final example with specifics:
  {topic['final_example']}
  ONLY brief clarifying follow-ups (1–3 sentences). Encourage "no doubts" to close.
- **ended**: Session closed (should not reach here).
{practice_hint}
{q_hint}

## Counters
Check questions so far: {session.questions_asked}/{len(questions)}
Practice stage: {session.practice_stage + 1 if all_stages else 0}/{len(all_stages) if all_stages else 0}
Complete follow-ups used: {session.complete_followups}/{MAX_COMPLETE_FOLLOWUPS}
"""


def _close_from_complete(session: Session, reason: str = "finished") -> str:
    topic = get_topic(session.topic_id)
    if reason == "finished":
        goodbye = (
            f"Awesome — sounds like you're all set on **{topic['name']}**! "
            "Session closed. Download your transcript anytime, or reset for a new topic."
        )
    else:
        goodbye = (
            f"We've covered your follow-up questions on **{topic['name']}**. "
            "Session closed — great work today!"
        )
    session.ended = True
    session.phase = "ended"
    return goodbye


def _update_phase(session: Session, user_message: str) -> None:
    topic = get_topic(session.topic_id)
    num_questions = len(topic["check_questions"])

    if session.phase == "reading" and RETURN_PATTERN.search(user_message):
        session.phase = "checking"
        session.questions_asked = 0
        session.awaiting_answer = False
        return

    if session.phase == "checking":
        if FORGOT_PATTERN.search(user_message):
            return
        if session.awaiting_answer and not IDK_PATTERN.search(user_message):
            session.questions_asked += 1
            session.awaiting_answer = False
        elif session.awaiting_answer and IDK_PATTERN.search(user_message):
            session.awaiting_answer = False
        if session.questions_asked >= num_questions or CONFIDENT_PATTERN.search(user_message):
            session.phase = "practice"
            session.practice_steps = 0
            session.practice_stage = 0
        return

    if session.phase == "practice":
        stages = topic.get("practice_stages", [])
        if session.awaiting_answer and not IDK_PATTERN.search(user_message):
            session.practice_steps += 1
            if stages:
                session.practice_stage = min(session.practice_stage + 1, len(stages))
            session.awaiting_answer = False
        elif session.awaiting_answer and IDK_PATTERN.search(user_message):
            session.awaiting_answer = False
        stage_complete = stages and session.practice_stage >= len(stages)
        if DONE_PATTERN.search(user_message) or stage_complete or (
            session.practice_steps >= len(stages) + 1
            and CONFIDENT_PATTERN.search(user_message)
        ):
            session.phase = "complete"
            session.complete_followups = 0
        return

    if session.phase == "complete" and not FINISH_PATTERN.search(user_message):
        session.complete_followups += 1
        return


def _should_ask_next(session: Session) -> bool:
    if session.phase in ("checking", "practice"):
        return True
    return False


def _idk_nudge(user_message: str, phase: str) -> str:
    if phase not in ("checking", "practice") or not IDK_PATTERN.search(user_message):
        return ""
    return (
        "\n\nStudent said they don't know. Respond with ONE everyday analogy or a simpler "
        "sub-question. Do NOT name vendor products or give the full answer. "
        "End by asking them to try again in their own words."
    )


def tutor_reply(session: Session, user_message: str) -> str:
    """Process a student message and return the tutor's reply."""
    if session.ended:
        raise ValueError("Session has ended")

    session.messages.append({"role": "user", "content": user_message})

    prev_phase = session.phase
    _update_phase(session, user_message)

    if session.phase == "complete" and prev_phase == "complete" and FINISH_PATTERN.search(
        user_message
    ):
        content = _close_from_complete(session, reason="finished")
        session.messages.append({"role": "assistant", "content": content})
        return content

    model = get_model(session.provider)
    system = _build_tutor_system(session)
    topic = get_topic(session.topic_id)
    questions = topic["check_questions"]

    extra = ""
    if prev_phase == "reading" and session.phase == "checking":
        first_q = questions[0] if questions else "What do you already know about this topic?"
        extra = (
            "\n\nThe student returned from reading. Welcome them briefly (one sentence). "
            f"Ask ONLY this first check question — freshman level, no product names:\n\"{first_q}\""
        )
    elif session.phase == "checking" and prev_phase == "checking":
        if session.questions_asked < len(questions):
            next_q = questions[session.questions_asked]
            extra = (
                f"\n\nBriefly acknowledge their last answer. Ask ONLY check question "
                f"{session.questions_asked + 1}/{len(questions)} — conceptual, no vendor trivia:\n"
                f"\"{next_q}\""
            )
        else:
            extra = "\n\nChecking is done. Transition to practice in one sentence."
    elif session.phase == "practice" and prev_phase == "checking":
        stage, _ = _practice_stage_info(session)
        stage = stage or get_topic(session.topic_id)["practice_stages"][0]
        extra = (
            f"\n\nMove to hands-on practice for {get_topic(session.topic_id)['practice_label']}. "
            f"Start with stage 1 — {stage['label']}: {stage['focus']} "
            "Ask ONE question. State the end goal briefly, then let the student lead."
        )
    elif session.phase == "practice" and prev_phase == "practice":
        stage, _ = _practice_stage_info(session)
        if stage:
            extra = (
                f"\n\nContinue practice at stage: {stage['label']}. {stage['focus']} "
                "If they jumped ahead in the wrong order, nudge them back gently."
            )
        else:
            extra = "\n\nPractice stages complete. Transition to wrap-up."
    elif session.phase == "complete" and prev_phase != "complete":
        extra = (
            "\n\nThe student finished practice. Congratulate them, share the final example "
            "concisely, and ask if they have any last doubts. Mention they can say "
            "'no doubts' when ready to close."
        )
    elif session.phase == "complete":
        extra = (
            "\n\nAnswer ONLY their follow-up briefly. Do not restart the lesson. "
            f"Follow-ups remaining before auto-close: "
            f"{max(0, MAX_COMPLETE_FOLLOWUPS - session.complete_followups)}."
        )

    extra += _idk_nudge(user_message, session.phase)

    response = model.invoke(
        [SystemMessage(content=system + extra)] + _history_for_llm(session.messages)
    )
    content = response.content if isinstance(response.content, str) else str(response.content)

    if _should_ask_next(session) and session.phase in ("checking", "practice"):
        session.awaiting_answer = True

    session.messages.append({"role": "assistant", "content": content})

    if session.phase == "complete" and session.complete_followups >= MAX_COMPLETE_FOLLOWUPS:
        close_note = _close_from_complete(session, reason="limit")
        session.messages.append({"role": "assistant", "content": close_note})

    return content


def latest_tutor_message(session: Session) -> str | None:
    for msg in reversed(session.messages):
        if msg["role"] == "assistant":
            return msg["content"]
    return None


def end_session(session: Session) -> str:
    """Manually close a session from complete or active phase."""
    if session.ended:
        return session.messages[-1]["content"]

    if session.phase == "complete":
        content = _close_from_complete(session, reason="finished")
    else:
        session.ended = True
        session.phase = "ended"
        topic = get_topic(session.topic_id)
        content = (
            f"Session closed early on **{topic['name']}**. "
            "You can download your transcript or reset to start a new topic."
        )

    session.messages.append({"role": "assistant", "content": content})
    return content
