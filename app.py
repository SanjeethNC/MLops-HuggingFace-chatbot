import gradio as gr
from huggingface_hub import InferenceClient

# Hugging Face Inference API client initialization
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    # Construct the conversation history including system message
    messages = [{"role": "system", "content": system_message}]
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    # Generate response from model with streaming
    response = ""
    for token_response in client.chat_completion(
        messages, max_tokens=max_tokens, stream=True, temperature=temperature, top_p=top_p
    ):
        token = token_response.choices[0].delta.content
        response += token
        yield response

# Function to reset chat history
def reset_chat():
    return "", []

# Custom CSS for modern UI styling
custom_css = """
#submit {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}
#reset {
    background-color: #f44336;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}
.gradio-container {
    background-color: #f0f4f8;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
"""

# Build the enhanced chatbot interface
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message", placeholder="Set the chatbot's system behavior."),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
        gr.Checkbox(value=True, label="Enable Markdown formatting"),
        gr.File(label="Upload previous chat history", type="filepath"),
    ],
    theme="soft",
    title="Enhanced AI Chatbot",
    description="A chatbot interface with advanced features and modern design.",
)

# Additional interface components (Reset, Save Chat, etc.)
with gr.Blocks(css=custom_css) as interface:
    gr.Markdown("# Welcome to the Enhanced Chatbot! 🌟")
    gr.Markdown("This chatbot comes with additional features like chat reset, chat history upload, and markdown support. Try adjusting the settings to enhance the conversation!")
    
    with gr.Row():
        with gr.Column(scale=3):
            demo.launch()
        with gr.Column(scale=1):
            reset_button = gr.Button("Reset Conversation", id="reset")
            reset_button.click(fn=reset_chat, inputs=[], outputs=[demo])

# Launch the interface with the custom layout
interface.launch()

if __name__ == "__main__":
    demo.launch()
