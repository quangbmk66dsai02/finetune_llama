# -*- coding: utf-8 -*-
from openai import OpenAI
from tqdm import tqdm
import os
import re 
import json
# Set the API key and model name
MODEL = "gpt-4o"
key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=key)

example_prompts=["""
Câu hỏi phức tạp 1:
1.Sự phát triển của cư dân thời tiền sử ở Việt Nam như thế nào?
2.Các câu hỏi đơn giản:
3.Cư dân ở Việt Nam thời kỳ đồ đá cũ sống như thế nào?
4.Đời sống và công cụ của người thời kỳ đồ đá mới ra sao?
""",
"""
Câu hỏi phức tạp 1:
Các hình thức cai trị của Trung Quốc đối với Việt Nam thay đổi như thế nào từ thời Hán đến thời Đường?
Các câu hỏi đơn giản:
1.Triệu Đà thành lập nước Nam Việt như thế nào?
2.Nam Việt bị nhà Hán thôn tính vào thời điểm nào?
3.Sự phản kháng của người Việt trước sự thôn tính của Hán ra sao?
4.Nhà Đường đã cai trị Việt Nam như thế nào?
 """,
"""
Câu hỏi phức tạp 1:
Sự biến chuyển trong hệ thống hành chính của Việt Nam từ thế kỉ 11 đến thế kỉ 19?

Các câu hỏi đơn giản:
1.Hệ thống hành chính của Việt Nam vào thế kỉ 11 – 13 có đặc điểm gì?
2.Có những thay đổi gì trong bộ máy hành chính từ thế kỉ 14 – 16?
3.Tổ chức hành chính Việt Nam thay đổi thế nào trong thế kỉ 17 – 18?
4.Hệ thống hành chính Việt Nam dưới triều Nguyễn (thế kỉ 19) có gì đặc biệt?""",

"""
Câu hỏi phức tạp 1:
Diễn biến và kết quả của cuộc kháng chiến chống Pháp (1945–1954) như thế nào?

Các câu hỏi đơn giản:
1.Tình hình Việt Nam sau Cách mạng tháng Tám và trước ngày toàn quốc kháng chiến?
2.Giai đoạn 1946–1949, cuộc chiến chống Pháp diễn ra như thế nào?
3.Những thay đổi chiến lược và lực lượng của ta từ 1950 đến 1953 là gì?
4.Chiến dịch Điện Biên Phủ năm 1954 có diễn biến và kết quả như thế nào?
""",
"""
Câu hỏi phức tạp 1:
Diễn biến và tác động của cuộc kháng chiến chống Mỹ (1954–1975)?

Các câu hỏi đơn giản:
1.Sau Hiệp định Genève năm 1954, tình hình chính trị hai miền như thế nào?
2.Giai đoạn 1959–1964, cuộc kháng chiến chống Mỹ phát triển ra sao?
3.Cuộc Tổng tiến công và nổi dậy Tết Mậu Thân 1968 có ý nghĩa gì?
4.Diễn biến và kết quả của chiến dịch Hồ Chí Minh năm 1975 là gì?
""",
"""
Câu hỏi phức tạp 1:
Sự phát triển kinh tế Việt Nam từ sau 1975 đến nay?

Các câu hỏi đơn giản:
1.Kinh tế Việt Nam sau năm 1975 phát triển như thế nào?
2.Giai đoạn Đổi Mới từ 1986 có những chuyển biến kinh tế nào nổi bật?
3.Trong thập niên 1990, Việt Nam đã hội nhập kinh tế thế giới ra sao?
4.Kinh tế Việt Nam từ năm 2000 đến nay phát triển theo xu hướng nào?

"""]
all_pairs = []
asked_questions = []  # new list to store all complex questions asked

topics = ["Tiền sử", "Bắc thuộc", "Phong kiến", "Kháng chiến chống Pháp thế kỷ 19 đến 1954", "Kháng chiến chống Mỹ 1954-1975", "Hiện đại"]
for i in range(0,6):
        topic = topics[i]
        example_prompt = example_prompts[i]
        print(f"Topic: {topic}")
        print(f"Example Prompt: {example_prompt}")
    # Call the API
        completion = client.chat.completions.create(
                model=MODEL,
                messages=[
            {"role": "system", "content": f"""Hãy nghĩ ra 20 câu hỏi phức tạp lịch sử Việt Nam (có thể phân tách theo tình tự thời gian được) về đề tài {topic} diễn ra trong 1 thời gian dài.
             Hãy phân tách với mỗi câu hỏi phức tạp thành 4 câu hỏi đơn giản theo trình tự thời gian.
                Đây là 1 ví dụ về hình thức và cách bạn có thể trả lời:
            {example_prompt}
                """,},
            {"role": "user", "content": f""" """},
                    ]
        )
        # Get the response
        response_content = completion.choices[0].message.content
        print("THIS IS THE RESPONSE CONTENT", "="*100, "\n", response_content)

        # Extract complex questions and their 4 subquestions
        # Extract complex questions and their 4 subquestions
        pattern = r"Câu hỏi phức tạp\s*\d*:\s*(.*?)\n+Các câu hỏi đơn giản:\n+1\.\s*(.*?)\n+2\.\s*(.*?)\n+3\.\s*(.*?)\n+4\.\s*(.*?)\n"
        matches = re.findall(pattern, response_content, re.DOTALL)

        for match in matches:
                complex_question = match[0].strip()
                subquestions = [f"Câu hỏi đơn giản {i+1}: {match[i+1].strip()}" for i in range(4)]
                all_pairs.append({
                        "question": complex_question,
                        "subquestions": subquestions
                        })
                asked_questions.append(complex_question)
# Save to JSON file
with open("data-for-complex/questions_time.json", "w", encoding="utf-8") as f:
    json.dump(all_pairs, f, ensure_ascii=False, indent=2)

# Save asked questions to a separate text file
with open("asked_questions.txt", "w", encoding="utf-8") as f:
    for q in asked_questions:
        f.write(q + "\n")

print(f"Saved {len(all_pairs)} question-subquestion pairs to questions.json")
print(f"Saved {len(asked_questions)} complex questions to asked_questions.txt")

# print("ASKED QUESTION", asked_questions)
# Log the response to a file

