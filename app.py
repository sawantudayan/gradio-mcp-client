import os

import gradio as gr
from smolagents import CodeAgent, InferenceClientModel
from smolagents.mcp_client import MCPClient

model = InferenceClientModel(token=os.getenv("HUGGINGFACE_API_TOKEN"))

try:
    mcp_client = MCPClient(
        {
            "url": "https://abidlabs-mcp-tools2.hf.space/gradio_api/mcp/sse"
        }
    )

    tools = mcp_client.get_tools()
    agent = CodeAgent(tools=[*tools], model=model, additional_authorized_imports=["json", "ast", "urllib", "base64"])


    def call_agent(message, history):
        return str(agent.run(message))


    demo = gr.ChatInterface(
        fn=call_agent,
        type="messages",
        examples=["Prime Factorization of 59"],
        title="MCP Client with Gradio",
        description="This is a simple agent that uses MCP Client to answer questions"
    )

except Exception as e:
    raise e
finally:
    mcp_client.disconnect()

if __name__ == "__main__":
    demo.launch(mcp_server=True)
