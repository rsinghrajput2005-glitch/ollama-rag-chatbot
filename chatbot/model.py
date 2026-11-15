from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

def final_output(filtered_docs, question):
    template = ChatPromptTemplate.from_messages([
        ("system", """
        You are a document-based question answering assistant.

        Rules:
        1. Answer ONLY using the provided context.
        2. Some changes is allowed to better answer.
        3. If the answer is not present in the context,
        respond exactly:
        "I don't know. This information is not available in the provided documents."

        Context:
        {content}
        """),
        ("human", "{question}")
    ])

    llm = ChatOllama(
        model="gemma3",
        temperature=0
    )

    chain = template | llm | StrOutputParser()

    if not filtered_docs:
        return {"message": "I don't know. Information not available in documents."}

    else:
        contents = "\n\n".join(
            doc.page_content for doc in filtered_docs
        )

        final = chain.invoke({
            "question": question,
            "content": contents
        })

        return final
