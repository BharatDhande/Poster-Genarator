import openai

def generate_image(prompt):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024",
        model="dall-e-3"
    )
    return response['data'][0]['url']
