from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# GPUの確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n!!! current device is {device} !!!\n")

# モデルのダウンロード
model_id = "inu-ai/dolly-japanese-gpt-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# LangChainのLLMとして利用
task = "text-generation"
pipe = pipeline(
    task, 
    model=model,
    tokenizer=tokenizer,
    device=f"{device}:0",
    framework='pt',
    max_new_tokens=20,
    temperature=0.1
)

# LLMs: langchainで上記モデルを利用する
llm = HuggingFacePipeline(pipeline=pipe)

# Prompts: プロンプトを作成
template = """<s>Question: {question} <SEP>
Answer: 段階的に考え回答する。同じ言葉の繰り返しを避け、要約すること。<\s>"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Chains: llmを利用可能な状態にする
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

# 質問を投げる
question = "Pythonでリストの最後尾を取得するには？"
generated_text = llm_chain.run(question)
print(generated_text)