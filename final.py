import gradio as gr
import os
from openai import OpenAI
from PIL import Image
import io
import tempfile
import pyttsx3
import base64
from transformers import pipeline
import numpy as np
from pydub import AudioSegment

import os
from twilio.rest import Client


account_sid = ""
auth_token = ""
twilio_client = Client(account_sid, auth_token)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# Initialize text-to-speech engine
engine = pyttsx3.init()
# transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def encode_image(img):
    image = img.resize((1000, 1000))  # Resize the image to a suitable resolution

    # Encode the image as base64
    with io.BytesIO() as output:
        image.save(output, format="JPEG")  # Convert image to JPEG format
        image_bytes = output.getvalue()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    return encoded_image

def analyze(text_input, image):
    encoded_image = encode_image(image)
    prompt = f"Input: {text_input}\n"
    response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {
        "role": "system", 
        "content": [{"type": "text", 
        "text": "You are a report generator. Generate very cleanly and very nicely and very evenly spaced formatted government reports."}]
        },
        {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            }
        ],
        },
        {
        "role": "user", 
        "content": [{"type": "text", 
        "text": "generate a demographic and weapon report based on this image. Return this in an organized and evenly spaced table. "}]
        },
    ],
    max_tokens=300,
    )
    ai_response_text = response.choices[0].message.content

    audioText = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {
        "role": "user", 
        "content": [{"type": "text", 
        "text": "You are a border defender"}]
        },
        {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            }
        ],
        },
        {
        "role": "user", 
        "content": [{"type": "text", 
        "text": "In a formal, intimidating, rough tone, give a basic description of the person/people in the image as if you were observing this, say how many people there are, say that we know the drone is in Tuscon, Arizona, say that we know the person is heading North, say that we know there is a firearm (if there is one in the image) and say that the person will be arrested if he does not turn around now."}]
        },
    ],
    max_tokens=300,
    )
    
    audio = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=audioText.choices[0].message.content
    )

    # Extract the audio content from the response
    audio_content = audio.content
    
    # Save audio content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_content)
        audio_file_path = f.name

    sound2 = AudioSegment.from_file("Police Siren Sound Effect.wav")
    sound1 = AudioSegment.from_file(audio_file_path)

    combined = sound1.overlay(sound2)

    combined.export("combined_audio.wav", format='wav')

    # ----------- Call
    phoneCallText = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            }
        ],
        },
        {
        "role": "user", 
        "content": [{"type": "text", 
        "text": "generate a description of the person and a weapon report based on this image if any is in the image. Do this as if to a phone call operator who will need to decide how to respond to the contents of the image."}]
        },
    ],
    max_tokens=300,
    )

    message = phoneCallText.choices[0].message.content
    call = twilio_client.calls.create(
        twiml=f'<Response><Say>{message}</Say></Response>',
        to="", # phone number
        from_="" # phone number
    )

    print(call.sid)

    return ai_response_text, "combined_audio.wav"

demo = gr.Interface(
    fn=analyze,
    inputs=[gr.Text(label = "Prompt"), gr.Image(label="Upload image", type="pil")],
    outputs=[gr.Text(label="Data Report"), gr.Audio(label="Scarecrow Audio")],
    description="Analyze Drone Images",
    allow_flagging="never",
    live=True,
)

demo.launch()