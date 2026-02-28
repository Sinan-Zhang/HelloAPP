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

# ===================== 全局模型初始化（只加载一次，避免重复） =====================
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# 全局缓存，避免每次生成都重新加载模型
@st.cache_resource(show_spinner="Loading story model...")
def load_story_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-small",
        device_map="auto"  # 自动分配CPU/GPU，适配Streamlit
    )
    return tokenizer, model

story_tokenizer, story_model = load_story_model()

# text2story
def text2story(text):
    # 基于老师指定的模型，优化生成参数+精准Prompt
    pipe = pipeline(
        "text-generation",
        model="pranavpsv/genre-story-generator-v2",
        # 核心参数优化：增加创意、防重复、控制字数
        model_kwargs={
            "temperature": 0.8,    # 增加故事创意和趣味性
            "top_p": 0.9,          # 提升内容多样性
            "repetition_penalty": 1.2,  # 禁止重复内容
            "max_length": 200,     # 控制故事总长度（对应80-120词）
            "min_length": 100,     # 保证字数足够
            "no_repeat_ngram_size": 2,  # 禁止2个词以上的重复
            "do_sample": True      # 生成更有创意的内容
        }
    )
    # 构造儿童向Prompt，明确要求生动、有角色/拟声词/情节
    prompt = f"""
    Write a fun, lively story for kids aged 3-10 based on this scene: {text}
    Requirements:
    1. 80-120 words (not too short!)
    2. Give cute names to characters (e.g., Lily, Tom, Mia)
    3. Add funny sound words (giggle, woof, splash, zoom)
    4. Include simple, happy plot (playing, making friends, adventure)
    5. Warm and happy ending
    6. No repeated sentences or boring phrases
    """
    # 生成故事并清理冗余内容
    story_text = pipe(prompt)[0]['generated_text']
    # 只保留Prompt之后的故事内容，去掉规则本身
    if "Requirements:" in story_text:
        story_text = story_text.split("Requirements:")[-1].strip()
    # 兜底：确保字数在80-120词
    story_words = story_text.split()
    if len(story_words) > 120:
        story_text = " ".join(story_words[:120]) + "!"
    elif len(story_words) < 80:
        story_text += " They laughed and played until the sun went down, promising to meet again tomorrow for more fun adventures!"
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
