from langchain.llms import HuggingFaceHub
import os

def get_model():
    model = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3", 
        model_kwargs={"temperature": 0.5}, 
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
    )
    return model

def create_prompt(context, question):
    prompt = f"""You are a helpful AI assistant. Based on the following context, answer the question. Given the case, there is not enough information to answer the query then apologize for the inconvinience caused and tell the user that not enough context is available.

                Context: {context}
                Question: {question}
                Answer:"""
    return prompt

def extract_answer(full_response, prompt):
    if "Answer:" in full_response:
        answer = full_response.split("Answer:")[-1].strip()
        return answer

    if full_response.startswith(prompt):
        answer = full_response[len(prompt):].strip()
        return answer

    return full_response

def get_response(context, question):
    prompt = create_prompt(context, question)
    model = get_model()
    full_response = model.invoke(prompt)
    answer = extract_answer(full_response, prompt)
    return answer
