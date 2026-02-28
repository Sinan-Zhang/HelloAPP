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
    # 新增：针对3-10岁儿童的趣味故事Prompt，明确生动性要求
    prompt = f"""
    Write a super fun story for kids aged 3-10 based on this scene: {text}
    Rules to make the story lively:
    1. Use simple words and short sentences (50-100 words total).
    2. Add cute characters with names (like Lily the rabbit, Tom the dog).
    3. Include funny sound words (like "woof woof", "giggle", "splash").
    4. Add simple dialogue between characters (e.g., "Let's play!", said Lily).
    5. Happy ending, warm and friendly tone.
    6. No hard words, no scary content.
    """
    # 优化：增加模型生成参数，提升故事创意和可控性
    pipe = pipeline(
        "text-generation",
        model="pranavpsv/genre-story-generator-v2",
        model_kwargs={"temperature": 0.8, "top_p": 0.9, "max_length": 200, "min_length": 50}
    )
    # 生成故事并清理冗余内容
    story_text = pipe(prompt)[0]['generated_text']
    story_text = story_text.replace(prompt, "").strip()
    # 兜底：确保字数在50-100词
    story_words = story_text.split()
    if len(story_words) > 100:
        story_text = " ".join(story_words[:100]) + "..."
    elif len(story_words) < 50:
        story_text += " They laughed and played all day, and became best friends forever! 🎉"
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
