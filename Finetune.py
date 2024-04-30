from openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] ="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"

client = OpenAI()
def set_api_key():
    OpenAI.api_key = "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"

def dataset_upload():
    result = client.files.create(
        file=open("/Users/erin/PycharmProjects/pythonProject/Dataset.jsonl", "rb"),
        purpose="fine-tune"
    )
    print(result)


def tune_now(uploaded_file_id):
    set_api_key()
    client.fine_tuning.jobs.create(
        training_file=uploaded_file_id,
        model="gpt-3.5-turbo"
    )

def chat_with_yx_model(model_id):
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a trading strategy and coding expert."},
            {"role": "user", "content": "Give me the code of a simple macd backtest code. Short moving average is 15, long moving average is 30. When short bigger than long, we hold, short smaller than long, we clean the position, let me type in the start, end date and the stock ticker."}
        ]
    )
    print(completion.choices[0].message)


if __name__ == '__main__':
    #step 1 : upload training dataset
    #dataset_upload()

    # step 2 : begin fine-tune ChatGPT model
    tune_now("file-5Ryj9zDu7A2LEHjJ43eqmCIW")

    # step 3 : use my trained model
    #chat_with_yx_model("ft:gpt-3.5-turbo-0613:personal::84NbaCOS" )