from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os

# Bước 1: Đường dẫn đến mô hình sau khi chuyển đổi
model_path = "D:/lm/lmstudio-community/gemma-3-1b-it-HF"

# Kiểm tra xem mô hình đã được chuyển đổi chưa
if not os.path.exists(model_path):
    print(f"❌ Không tìm thấy mô hình tại: {model_path}")
    print("⚠️ Hãy chắc chắn rằng bạn đã chuyển đổi mô hình sang định dạng Hugging Face.")
    exit(1)

# Bước 2: Tải mô hình và tokenizer
try:
    print("🔄 Đang tải mô hình và tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("✅ Tải mô hình thành công!")
except Exception as e:
    print(f"❌ Lỗi khi tải mô hình: {e}")
    exit(1)

# Bước 3: Tải dữ liệu huấn luyện
print("🔄 Đang tải dữ liệu...")
try:
    dataset = load_dataset("json", data_files="dataset.json")
    print("✅ Dữ liệu tải thành công!")
except Exception as e:
    print(f"❌ Lỗi khi tải dữ liệu: {e}")
    exit(1)

# Bước 4: Xử lý dữ liệu (tokenize)
def tokenize_function(example):
    return tokenizer(example["prompt"] + tokenizer.eos_token + example["response"], truncation=True)

try:
    print("🔄 Đang tokenize dữ liệu...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print("✅ Tokenize dữ liệu thành công!")
except Exception as e:
    print(f"❌ Lỗi khi tokenize dữ liệu: {e}")
    exit(1)

# Bước 5: Cấu hình tham số huấn luyện
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy="epoch",
    save_total_limit=3,
)

# Bước 6: Khởi tạo Trainer
try:
    print("🔄 Khởi tạo Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )
    print("✅ Trainer khởi tạo thành công!")
except Exception as e:
    print(f"❌ Lỗi khi khởi tạo Trainer: {e}")
    exit(1)

# Bước 7: Huấn luyện mô hình
try:
    print("🚀 Bắt đầu huấn luyện...")
    trainer.train()
    print("✅ Huấn luyện thành công!")
except Exception as e:
    print(f"❌ Lỗi khi huấn luyện: {e}")
    exit(1)

# Bước 8: Lưu mô hình sau khi fine-tune
try:
    print("💾 Đang lưu mô hình...")
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("✅ Mô hình đã được lưu thành công!")
except Exception as e:
    print(f"❌ Lỗi khi lưu mô hình: {e}")
    exit(1)
