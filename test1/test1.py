import requests
import json

# URL API cho phần hội thoại
url = "http://192.168.56.1:1234/v1/chat/completions"

# Kiểm tra kết nối mô hình
def check_model():
    model_url = "http://192.168.56.1:1234/v1/models"
    response = requests.get(model_url)
    if response.status_code == 200:
        models = response.json()
        if len(models) > 0:
            print("✅ Mô hình khả dụng:", models)
        else:
            print("⚠️ Không có mô hình nào được khởi chạy.")
    else:
        print(f"❌ Lỗi kết nối: {response.status_code}")

# Gọi hàm kiểm tra mô hình
check_model()

# Bắt đầu hội thoại
print("\n💬 Bắt đầu trò chuyện với LM Studio. Gõ 'exit' để thoát.")
messages = []

while True:
    user_input = input("👤 Bạn: ")
    if user_input.lower() in ["exit", "thoát"]:
        print("👋 Kết thúc trò chuyện.")
        break

    # Thêm câu hỏi của người dùng vào danh sách hội thoại
    messages.append({"role": "user", "content": user_input})

    # Payload gửi lên API
    payload = {
        "model": "llama2-7b-chat",  # Thay bằng mô hình bạn đã chọn
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.7
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Gửi yêu cầu POST đến LM Studio
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            print(f"🤖 LM Studio: {reply}")
            # Thêm câu trả lời của mô hình vào hội thoại để giữ ngữ cảnh
            messages.append({"role": "assistant", "content": reply})
        else:
            print(f"❌ Lỗi kết nối: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Lỗi khi gửi yêu cầu: {e}")
