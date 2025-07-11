from openai import OpenAI
from openai import RateLimitError, OpenAIError

my_key = ""
client = OpenAI(api_key=my_key)


try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",  # 必须是正确支持的聊天模型
        messages=[
            {"role": "user", "content": "写一句关于独角兽的睡前故事，请用中文回答。"}
        ]
    )
    print(response.choices[0].message.content)
except RateLimitError:
    print("调用频率或额度已超出，请检查API Key的使用情况。")
except OpenAIError as e:
    print(f"调用失败：{e}")


