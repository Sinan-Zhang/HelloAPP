# Program title: Storytelling App

# import part
import streamlit as st
from transformers import pipeline

# function part
# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story
def text2story(text):
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch

    # 用更稳定的轻量模型
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 强化版 Prompt：明确要求角色、动作、拟声词和快乐结局
    prompt = f"""
    Write a lively, fun story for kids aged 3-10 based on this picture: {text}
    Follow these rules strictly:
    1. 50-100 words only.
    2. Give kids cute names (like Lily, Tom, Mia).
    3. Add funny sound words (giggle, woof, splash, zoom).
    4. Include simple actions (skipping, kicking, building).
    5. Happy, warm ending.
    6. No repeated phrases like "kids love to play".
    Only return the story, no extra words.
    """

    # 生成故事
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=150,
        min_length=50,
        temperature=0.8,  # 增加创意
        top_p=0.9,
        do_sample=True,
        num_beams=4
    )
    story_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 兜底：确保字数和趣味性
    story_words = story_text.split()
    if len(story_words) < 50:
        story_text += " They laughed until their tummies hurt, and promised to come back tomorrow!"
    elif len(story_words) > 100:
        story_text = " ".join(story_words[:100]) + "..."
    return story_text

# text2audio
def text2audio(story_text):
    pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = pipe(story_text)
    return audio_data


def main():
    st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
    st.header("Turn Your Image to Audio Story")
    uploaded_file = st.file_uploader("Select an Image...")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)


        #Stage 1: Image to Text
        st.text('Processing img2text...')
        scenario = img2text(uploaded_file.name)
        st.write(scenario)

        #Stage 2: Text to Story
        st.text('Generating a story...')
        story = text2story(scenario)
        st.write(story)

        #Stage 3: Story to Audio data
        st.text('Generating audio data...')
        audio_data =text2audio(story)

        # Play button
        if st.button("Play Audio"):
            # Get the audio array and sample rate
            audio_array = audio_data["audio"]
            sample_rate = audio_data["sampling_rate"]

            # Play audio directly using Streamlit
            st.audio(audio_array,
                     sample_rate=sample_rate)


if __name__ == "__main__":
    main()
