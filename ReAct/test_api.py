# Webサーバ (notebook外で実行)
from flask import Flask, request
import hashlib

app = Flask(__name__)

# チャット・メールなどを含む社内情報を収集し、トピックごとに分類されたデータをWebサーバが保持していると仮定。

topics = {
    "プロジェクト管理の効率化" : "チームのプロジェクト管理プロセスを改善し、効率的なタスク管理とスケジュール管理を実現するための施策は、ｘｘｘｘｘｘ",
    "技術トレンドの研究と導入" : "最新の技術トレンドを追い、適切な案件においてその導入を検討し、競争力を高める。現状はｙｙｙｙｙｙ",
    "チームメンバーのスキル開発": "チームメンバーのスキル開発を促進し、トレーニングや教育プログラムの導入により技術力と専門知識の向上を図る。現在、ｚｚｚｚｚｚ",
}

@app.route('/api/list_topics', methods=['GET'])
def list_topics():
    return list(topics.keys())

@app.route('/api/get_topic', methods=['POST'])
def get_topic():
    topic_name = request.form.get('topic_name')
    topic = topics[topic_name]
    
    return topic 
     
if __name__ == '__main__':
    app.run()