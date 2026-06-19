from dotenv import load_dotenv

from langchain_groq import ChatGroq

from src.prompts import (
    get_question_prompt,
    get_evaluation_prompt
)

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)


def generate_questions(
    context,
    difficulty
):

    prompt = get_question_prompt()

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "difficulty": difficulty
        }
    )

    return response.content


def evaluate_answer(
    question,
    answer
):

    prompt = get_evaluation_prompt()

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "answer": answer
        }
    )

    return response.content