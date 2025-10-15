import google.generativeai as genai



genai.configure(api_key="AIzaSyDwpFNc_TngScCnscZC75mTgcjfhMp8-Do")

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [

]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[
])
user = input("Enter your text")
convo.send_message(user)
print(convo.last.text)