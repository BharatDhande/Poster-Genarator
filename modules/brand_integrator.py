from PIL import Image

def add_branding(poster_path):
    poster = Image.open(poster_path)
    logo = Image.open("assets/logo.png").resize((100, 100))
    poster.paste(logo, (poster.width - 120, poster.height - 120), logo)
    poster.save("output/poster_branded.png")
    return "output/poster_branded.png"
