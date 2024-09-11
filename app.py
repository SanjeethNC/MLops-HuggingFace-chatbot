import gradio as gr
from huggingface_hub import InferenceClient

# Initialize Hugging Face Inference API client
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
print("asjaj")
print('abc')

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    *args,  # To catch additional arguments that may be passed
    **kwargs  # Catch any other keyword arguments Gradio may pass
):
    # Construct the conversation history including system message
    messages = [{"role": "system", "content": system_message}]
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    
    # Add the latest user message to the conversation
    messages.append({"role": "user", "content": message})

    response = ""
    # Stream the response from the model
    for token_response in client.chat_completion(
        messages, max_tokens=max_tokens, stream=True, temperature=temperature, top_p=top_p
    ):
        token = token_response.choices[0].delta.content
        response += token
        yield response

# Custom CSS for better UI appearance
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

# Interface setup with chat input, sliders, and file upload
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
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

# Launch the demo with the option to share a public link
if __name__ == "__main__":
    demo.launch(share=True)
