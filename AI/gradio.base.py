import gradio as gr

def echo(message, history):
    return message

demo = gr.ChatInterface(fn=echo, type="messages", examples=["不死三振是甚麼?", "強迫進壘定義為何?", "高飛必死球是甚麼意思?"], title="九局通")
demo.launch()