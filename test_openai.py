from openai import OpenAI
import os

# Load API key from environment safely
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("Testing OpenAI API…")

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Give me a plant care tip for tomatoes in one sentence."}
        ]
    )

    print("\n✅ API Test Successful")
    print("Response:", response.choices[0].message.content)

except Exception as e:
    print("\n❌ API Test Failed")
    print("Error:", e)
