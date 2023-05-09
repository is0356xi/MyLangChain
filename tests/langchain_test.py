from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Prompts: プロンプトを作成
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Models(LLMs): 変換後のモデルを読み込み
local_path = "/Downloads/gpt4all/models/ggml-model-q4_0.bin"
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(
    model=local_path, 
    callbacks=callbacks, 
    verbose=True,
    streaming=True
)

# Chains: 作成したモデルを利用可能な状態にする
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 質問を投げる
question = "How can I get the end of the list in Python?"
llm_chain.run(question)