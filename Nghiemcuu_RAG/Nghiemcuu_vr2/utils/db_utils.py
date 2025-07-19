import os, pymysql
from dotenv import load_dotenv

load_dotenv()

def _conn(schema: str):
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=schema,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )

def question_exists(cursor, question_text):
    query = "SELECT COUNT(*) FROM questions WHERE question_text = %s"
    cursor.execute(query, (question_text,))
    return cursor.fetchone()[0] > 0

def insert_question(cursor, question_text, answer_text, source_file):
    query = "INSERT INTO questions (question_text, answer_text, source_file) VALUES (%s, %s, %s)"
    cursor.execute(query, (question_text, answer_text, source_file))


# Lưu vào cơ sở dữ liệu
def save_qas(schema: str, qas: list[tuple[str, str]], category="general"):
    sql = """
    INSERT INTO qa_entries (question, answer, category)
    VALUES (%s,%s,%s)
    ON DUPLICATE KEY UPDATE
        answer = VALUES(answer),
        updated_at = CURRENT_TIMESTAMP
    """
    with _conn(schema) as conn:
        with conn.cursor() as cur:
            for q, a in qas:
                cur.execute(sql, (q, a, category))
        conn.commit()