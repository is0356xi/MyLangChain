import requests

# OpenAI ChatのURL
url = "https://chat.openai.com/"

# セッションを開始
session = requests.Session()
session.get(url)

print(session)

# ログインフォームから_csrfトークンを取得
login_page = session.get(url)
csrf_token = login_page.cookies['csrf_token']

# メッセージの内容
message = "Hello, OpenAI Chat!"

# メッセージの送信
response = session.post(
    url + "/api/send_message",
    headers={
        "X-CSRFToken": csrf_token,
        "Referer": url,
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest"
    },
    json={"text": message}
)

# レスポンスの表示
print(response.json())