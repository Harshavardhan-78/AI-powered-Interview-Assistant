from langchain_core.prompts import ChatPromptTemplate


def get_question_prompt():

    return ChatPromptTemplate.from_template(
        """
        You are an experienced technical interviewer.

        Context:
        {context}

        Generate 5 {difficulty} level interview questions
        based only on the candidate profile.

        Return only questions.
        """
    )


def get_evaluation_prompt():

    return ChatPromptTemplate.from_template(
        """
        You are a senior interviewer.

        Question:
        {question}

        Candidate Answer:
        {answer}

        Evaluate the answer.

        Give:

        Score (out of 10)

        Strengths

        Weaknesses

        Improved Answer
        """
    )