import openai

def generate_caption(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Write a social media caption for: {prompt}"}]
    )
    return response['choices'][0]['message']['content']
