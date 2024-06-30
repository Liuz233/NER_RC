import json
from time import sleep

import requests

# 配置API相关信息
host = 'https://llmaiadmin-test.classba.cn'
authorization_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MzYsImlhdCI6MTcxOTA1NTgyOCwiZXhwIjoxNzIxNjQ3ODI4fQ.atmTOzXbkTN7Z0jX8QsC1GZO-sJQqv6Av7Nbu-rnCVc'
api_key_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRhIjoiMGFjOTQ2ZjhmMTEyNDUwODZhZDdhZWM0ODZjYjZiNWMiLCJzdGF0dXMiOjB9.bgkSmARpQt0Zy5rBmrZgc9IaZ3hl83eeCeSjrG7E82A'
session_id = 123

# 新建会话
start_session_url = f'{host}/api/bot/start'
start_session_headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {authorization_token}',
    'apikeytoken': api_key_token
}
start_session_data = {
    'id': session_id
}

start_response = requests.post(start_session_url, headers=start_session_headers, json=start_session_data, timeout=10)
start_response_data = start_response.json()

if start_response_data['code'] != 0:
    raise Exception(f"Failed to start session: {start_response_data['msg']}")

session_key = start_response_data['data']['key']
print(f"Session started successfully with session key: {session_key}")

send_message_url = f'{host}/api/bot/chat'
send_message_headers = start_session_headers


# 发送消息 (流式)
def stream_messages(url, headers, data, retries=100):
    for attempt in range(retries):
        try:
            with requests.post(url, headers=headers, json=data, stream=True, timeout=10) as response:
                if response.status_code != 200:
                    raise Exception(f"Failed to stream messages: {response.status_code}")
                result = ""
                for line in response.iter_lines():
                    if line:
                        # 处理流式响应的每一行
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            temp = decoded_line[6:]
                            if temp != "{}":
                                result += temp
                return result
        except requests.exceptions.RequestException as e:
            print(f"Error during streaming message: {e}. Retrying ({attempt + 1}/{retries})...")
            sleep(1)  # 重试前等待片刻
    raise Exception("Failed to stream messages after multiple retries.")

def unstream_messages(url, headers, data, retries=100):
    for attempt in range(retries):
        try:
            with requests.post(url, headers=headers, json=data, stream=True, timeout=10) as response:
                if response.status_code != 200:
                    raise Exception(f"Failed to stream messages: {response.status_code}")
                result = ""
                return result
        except requests.exceptions.RequestException as e:
            print(f"Error during streaming message: {e}. Retrying ({attempt + 1}/{retries})...")
            sleep(1)  # 重试前等待片刻
    raise Exception("Failed to stream messages after multiple retries.")

# 读取输入文件
input_file = 'data/SemEval2010_task8/test.txt'
output_file = 'output.txt'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for i, line in enumerate(infile):
        sleep(0.1)
        line = line.strip()
        if line:
            stream_message_data = {
                'session_key': session_key,
                'msg': f"{line}",
                'stream': False
            }

            response_text = unstream_messages(send_message_url, send_message_headers, stream_message_data)

            # 解析API响应，提取两个最有可能的实体（假设API返回的是JSON格式）
            entities = response_text
            print(f"{i}:{entities}")

            # 将实体写入输出文件
            outfile.write(''.join(entities) + '\n')
