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

example_prompts=[
"""
Câu hỏi phức tạp 1:
Đại tướng Võ Nguyên Giáp sinh ở đâu?
Các câu hỏi đơn giản:
1. Đại tướng Võ Nguyên Giáp sinh ra ở đâu?
2. Đại tướng Võ Nguyên Giáp sinh ra vào thời gian nào?
""",
"""
Câu hỏi phức tạp 1:
Nhà Hán xâm lược Đại Việt vào lúc nào?
Các câu hỏi đơn giản:
1. Nhà Hán xâm lược Đại Việt vào lúc nào?
""",
"""
Câu hỏi phức tạp 1:
Phân tích vai trò của công nghiệp, nông nghiệp và dịch vụ trong phát triển kinh tế Việt Nam từ 1975 đến nay.

Các câu hỏi đơn giản:
1. Ngành công nghiệp Việt Nam sau 1975 phát triển như thế nào?
2. Nông nghiệp đã có những cải cách gì quan trọng?
3. Ngành dịch vụ đóng vai trò ra sao trong nền kinh tế hiện đại?
"""
]
all_pairs = []
asked_questions = []  # new list to store all complex questions asked

topics = ["Tiền sử", "Bắc thuộc", "Phong kiến", "Kháng chiến chống Pháp thế kỷ 19 đến 1954", "Kháng chiến chống Mỹ 1954-1975", "Hiện đại"]
concat_prompt = ""
for i in range(6):
      concat_prompt += example_prompts[i] 
      concat_prompt += "\n"
for i in range(0,6):
        topic = topics[i]
        # example_prompt = example_prompts[i]
        example_prompt = concat_prompt
        # print(f"Topic: {topic}")
        # print(f"Example Prompt: {example_prompt}")
    # Call the API
        completion = client.chat.completions.create(
                model=MODEL,
                messages=[
            {"role": "system", "content": f"""Hãy nghĩ ra 30 câu hỏi phức tạp lịch sử Việt Nam có nhiều chủ thể trong câu hỏi về đề tài {topic} diễn ra trong 1 thời gian dài.
             Hãy phân tách với mỗi câu hỏi phức tạp thành các câu hỏi đơn giản, mỗi 1 câu hỏi con ứng với 1 chủ thể.
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
        pattern = r"Câu hỏi phức tạp\s*\d*:\s*(.*?)\n+Các câu hỏi đơn giản:\n+(1\..*?)(?:\n+2\..*?)?(?:\n+3\..*?)?(?:\n+4\..*?)?(?=\n*Câu hỏi phức tạp|\Z)"

        # Use a broader match and then post-process each block
        block_pattern = r"Câu hỏi phức tạp\s*\d*:\s*(.*?)\n+Các câu hỏi đơn giản:\n+(.*?)(?=\n*Câu hỏi phức tạp|\Z)"
        matches = re.findall(block_pattern, response_content, re.DOTALL)

        for complex_q, subq_block in matches:
            subq_lines = re.findall(r"\d+\.\s*(.*)", subq_block.strip())
            subq_formatted = [f"Câu hỏi đơn giản {i+1}: {line.strip()}" for i, line in enumerate(subq_lines)]

            all_pairs.append({
                "question": complex_q.strip(),
                "subquestions": subq_formatted
            })
            asked_questions.append(complex_q.strip())

# Save to JSON
with open("parsed_questions.json", "w", encoding="utf-8") as f:
    json.dump(all_pairs, f, ensure_ascii=False, indent=2)

# Save complex questions only
with open("parsed_asked_questions.txt", "w", encoding="utf-8") as f:
    for q in asked_questions:
        f.write(q + "\n")

print(f"✅ Saved {len(all_pairs)} complex questions and subquestions.")

# print("ASKED QUESTION", asked_questions)
# Log the response to a file

