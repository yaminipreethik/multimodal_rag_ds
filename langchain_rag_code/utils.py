import base64

def display_base64_image(base64_code):
    from IPython.display import Image, display
    image_data = base64.b64decode(base64_code)
    display(Image(data=image_data))
