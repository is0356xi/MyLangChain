from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from langchain.memory import ConversationBufferMemory
import torch

def model_setup(model_id:str):
    # モデル&トークナイザーのダウンロード
    print(f"!!! Downloading Model from {model_id} !!!")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    return model, tokenizer


def pipeline_setup(model, tokenizer, isGPU:bool, **kwargs) -> HuggingFacePipeline:
    # GPUの確認
    if isGPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n!!! current device is {device} !!!")
        model = model.to(device)
        
        # GPUにモデルを展開する際に必要な引数を追加
        device = 0
        framework = 'pt'
    else:
        device = -1
        framework = None
        
        
    # パイプラインの作成
    task = "text-generation"
    pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        device=device,
        framework=framework,
        pad_token_id=0,
        **kwargs
    )

    # LLMs: LangChainで利用可能な形に変換
    llm = HuggingFacePipeline(pipeline=pipe)
    
    print("!!! Pipeline Setup Completed !!!\n\n")
    
    return llm



# Stopの条件を設定するクラスを作成 (StoppingCriteriaを継承する)
class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_str, num_iter, tokenizer, isGPU):
        if isGPU:
            self.stop_token_ids = tokenizer(stop_str, return_tensors='pt')["input_ids"].to('cuda')
            self.stop_token_ids_iter = tokenizer(stop_str*2, return_tensors='pt')["input_ids"].to('cuda')
        else:
            self.stop_token_ids = tokenizer(stop_str, return_tensors='pt')["input_ids"]
            self.stop_token_ids_iter = tokenizer(stop_str, return_tensors='pt')["input_ids"]
            
        self.num_iter = num_iter
        self.tokenizer = tokenizer
        
    def __call__(self, input_ids:torch.LongTensor, score:torch.FloatTensor, **kwargs):
        # 出力の最後尾の文字列とstop_strが一致した回数
        match_count = 0
        
        # 出力文字列を最後尾から順に、num_iterで指定された要素数だけ処理する
        for i in range(1, self.num_iter+1): 
            input_id = input_ids[0][-i]
            stop_id = self.stop_token_ids[0][0]
            stop_iter_id = self.stop_token_ids_iter[0][0]
            
            # 対象文字列とstop_strが一致した場合、カウントを増やす
            if input_id == stop_id:
                match_count += 1
            
        # \nが2回続いた場合、または\n\nが現れた場合、generate()をStopする
        if match_count == self.num_iter or input_id == stop_iter_id:
            isStop = True
            # print(f"!!! Generate() Stopped !!!\n!!!!!!!!!\n{self.tokenizer.decode(input_ids[0])} \n!!!!!!!!!")
        else:
            isStop = False
        return isStop
    
    
def chat_chain_setup(template, llm) -> LLMChain:
    # Memory: メモリ上に会話を記録する設定
    memory_key = "chat_history"
    memory = ConversationBufferMemory(memory_key=memory_key, ai_prefix="")
    
    # Prompts: プロンプトを作成
    prompt = PromptTemplate(template=template, input_variables=["chat_history", "input"])

    # Chains: プロンプト&モデル&メモリをチェーンに登録
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )
    
    return llm_chain


def main(isGPU=False):
    # モデルをダウンロード
    model_id = "andreaskoepf/pythia-1.4b-gpt4all-pretrain"
    model, tokenizer = model_setup(model_id)

    # Stopの条件式に用いる文字と、その文字が何回続いたらStopするかを指定
    stop_str = "\n"
    num_iter = 2  # \nが2回繰り返された場合、generate()をstopする

    # StoppingCriteriaListクラスのインスタンスを生成
    stopcriteria_list = StoppingCriteriaList([MyStoppingCriteria(stop_str, num_iter, tokenizer, isGPU=True)])
    print(stopcriteria_list)

    # HuggingFacePipelineを作成
    model_args = {"temperature":0.1, "max_length": 256, "stopping_criteria": stopcriteria_list}
    llm = pipeline_setup(model=model, tokenizer=tokenizer, isGPU=isGPU, **model_args)

    # プロンプトテンプレートを作成
    template = """
You are an AI who responds to user Input.
Please provide an answer to the human's question.
Additonaly, you are having a conversation with a human based on past interactions.

### Answer Sample
Human: Hi!
AI: Hi, nice to meet you.

### Past Interactions
{chat_history}

### 
Human:{input}
"""

    # Chat用のチェーンを作成
    llm_chain = chat_chain_setup(template, llm)

    # チャット形式
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break
        else:
            response = llm_chain.predict(input=user_input)
            print(response)


if __name__ == "__main__":
    import sys
    try:
        isGPU = bool(sys.argv[1])
    except Exception as e:
        print(f"{str(e)}: You are using CPU")
        isGPU = False

    main(isGPU)