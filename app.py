import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
import gradio as gr

# Respond function (you can customize it as per your use case)
def respond(message):
    # Example response logic, modify this as needed
    return "You said: " + message

# Custom CSS for enhancing the UI look
css = """
#chatbox {
    background-color: #f7f7f7;
    border-radius: 15px;
}
#submit {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}
#reset {
    background-color: #f44336;
    color: white;
    font-weight: bold;
}
.gradio-container {
    font-family: 'Arial', sans-serif;
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
"""

# Interface with multiple customizable settings
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message", placeholder="Set system behavior here..."),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
        gr.Checkbox(value=True, label="Enable Markdown formatting"),
        gr.File(label="Upload chat history", type="file"),
    ],
    theme="soft",
    title="My Cool Chatbot",
    description="A chatbot interface with modern features.",
    reset_on_submit=True,
    live=True,
)

# Additional Gradio components for enhancing the UI
with gr.Blocks(css=css) as interface:
    gr.Markdown(
        """
        # Welcome to the Cool Chatbot! 
        Let's have a friendly chat, and feel free to tweak the settings on the left to improve our conversation!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            demo.launch()
        with gr.Column(scale=1):
            gr.Button("Reset Conversation", id="reset")

# Launch the app with the custom layout
interface.launch()



if __name__ == "__main__":
    demo.launch()