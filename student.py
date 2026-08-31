"""Simulated student — only used when the user clicks the button."""

from langchain_core.messages import SystemMessage

from models import get_model
from personalities import get_personality_prompt
from topics import get_topic
from tutor import Session, _practice_stage_info, latest_tutor_message


def simulate_student_reply(session: Session) -> str:
    tutor_msg = latest_tutor_message(session)
    if not tutor_msg:
        return "Ready — I read through the links."

    personality_prompt = get_personality_prompt(session.personality)
    model = get_model(session.provider)

    stage_hint = ""
    if session.phase == "practice":
        stage, stages = _practice_stage_info(session)
        if stage:
            order = " → ".join(s["label"] for s in stages)
            stage_hint = (
                f"\nPractice stage now: {stage['label']}. Build order: {order}. "
                "If overconfident, you might jump ahead — otherwise stay on the current step."
            )

    topic_name = get_topic(session.topic_id)["name"]

    system = f"""You are simulating an undergrad student in a self-directed tutoring chat.

## Personality
{personality_prompt}

## Topic
{topic_name}

## Tutor just said
\"\"\"
{tutor_msg}
\"\"\"

## Rules
- Reply in first person as the student ONLY (1–3 short sentences).
- If the tutor asked you to read links, signal you're ready in varied ways like
  ready, I read it, let's go, or ok I'm ready — not always the same phrase.
- If the tutor asks an eliciting question, attempt YOUR answer first — don't ask the tutor to explain.
- If confused / missing context: partial answers, ask for clarification or a one-line reminder.
- If overconfident: answer quickly, maybe skip a step or use a buzzword incorrectly.
- If the lesson is complete, ask ONE short follow-up OR say 'no doubts' / 'all good'.
- NEVER copy the tutor's message. NEVER teach or explain like a tutor.
- NEVER write full diagrams, SRS documents, or long lists.

## Session phase: {session.phase}{stage_hint}
"""

    response = model.invoke([SystemMessage(content=system)])
    content = response.content if isinstance(response.content, str) else str(response.content)
    return content.strip()
