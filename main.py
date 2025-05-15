import gradio as gr
from llama_cpp import Llama

llm = Llama(
    model_path="models/phi-2.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
    n_batch=512,
    verbose=True
)

def chat(user_message, history):
    prompt = ""
    for human, assistant in history:
        prompt += f"Q: {human}\nA: {assistant}\n"
    prompt += f"Q: {user_message}\nA:"

    response = llm(
        prompt,
        temperature=0.2,
        max_tokens=300,
        stop=["Q:"]
    )
    bot_message = response["choices"][0]["text"].strip()

    return bot_message  # Only return assistant reply

gr.ChatInterface(
    fn=chat,
    title="Off-Chatty",
    description="Ask anything you want! Powered by Phi-2 running fully offline.",
    theme="soft",
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Type your message..."),
).launch()
