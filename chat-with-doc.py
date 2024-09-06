import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import tkinter as tk
from tkinter import scrolledtext

#os.environ["OPENAI_API_KEY"] = "{YOURAPIKEY}"
os.environ["OPENAI_API_KEY"] = "YOUR KEY"
# You MUST add your PDF to local files in this notebook (folder icon on left hand side of screen)

# Simple method - Split by pages
loader = PyPDFLoader("./CAF-Introduction.pdf")
pages = loader.load_and_split()
#print(pages[0])

# SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
chunks = pages
# Advanced method - Split by chunk

# Step 1: Convert PDF to text
import textract
doc = textract.process("./CAF-Introduction.pdf")

# Step 2: Save to .txt and reopen (helps prevent issues)
with open('CAF-Introduction.txt', 'w',encoding='utf-8') as f:
    f.write(doc.decode('utf-8'))

with open('CAF-Introduction.txt', 'r',encoding='utf-8') as f:
    text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])


# Quick data visualization to ensure chunking was successful

# Create a list of token counts
token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

# Create a DataFrame from the token counts
df = pd.DataFrame({'Token Count': token_counts})

# Create a histogram of the token count distribution
df.hist(bins=40, )

# Show the plot
plt.show()

# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)


query = "Who created transformers?"
docs = db.similarity_search(query)
docs[0]


chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
#chain = load_qa_chain(OpenAI(model="gpt-3.5-turbo", temperature=0), chain_type="stuff")

query = "Who created transformers?"
docs = db.similarity_search(query)

chain.run(input_documents=docs, question=query)


#from IPython.display import display
#import ipywidgets as widgets

# Create conversation chain that uses our vectordb as retriver, this also allows for chat history management
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())
#qa = ConversationalRetrievalChain.from_llm(OpenAI(model="gpt-3.5-turbo", temperature=0.1), db.as_retriever())

chat_history = []


def submit_query():
    user_input = input_text.get("1.0", tk.END).strip()
    if user_input.lower() == 'exit':
        root.destroy()
        return
    result = qa({"question": user_input, "chat_history": chat_history})
    chat_history.append((user_input, result['answer']))

    chat_display.config(state=tk.NORMAL)
    chat_display.insert(tk.END, f"You: {user_input}\n")
    chat_display.insert(tk.END, f"Bot: {result['answer']}\n\n")
    chat_display.config(state=tk.DISABLED)
    input_text.delete("1.0", tk.END)

# Start the conversation loop
#handle_input()

root = tk.Tk()
root.title("Chatbot Interface")

# Create chat display area
chat_display = scrolledtext.ScrolledText(root, state=tk.DISABLED, wrap=tk.WORD, height=20, width=80)
chat_display.pack(padx=10, pady=10)

# Create input text area
input_text = tk.Text(root, height=4, width=80)
input_text.pack(padx=10, pady=(0, 10))

# Create submit button
submit_button = tk.Button(root, text="Submit", command=submit_query)
submit_button.pack(pady=10)

# Run the application
root.mainloop()
