from dotenv import load_dotenv

from langchain_core.output_parsers import JsonOutputParser

# 1. 환경 변수 로드
load_dotenv()

# 2. JSON 출력 파서 생성
parser = JsonOutputParser()

# 3. 모델에게 줄 JSON 형식 지침 얻기
instructions = parser.get_format_instructions()

# 4. 출력
print(instructions)

print("----- JSON 파싱 테스트 -----")

# 4. AI가 JSON 문자열을 반환했다고 가정
ai_response = '{"이름": "김철수", "나이": 30}'

# 5. JSON 문자열 → 파이썬 dict 로 변환
parsed_response = parser.parse(ai_response)

# 6. 결과 출력
print(parsed_response)