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

# ===================== 修复后的 text2story 函数 =====================
def text2story(text):
    # 【关键修复】指令式Prompt：T5模型最吃这一套，明确要求NO REPETITION
    prompt = """
    Generate a fun story for 3-10 year olds about: {}.
    Rules:
    1. 50-100 words.
    2. Use names like Leo, Mia, or Zara.
    3. Add sound words (giggle, zoom, splash).
    4. NO REPEATED SENTENCES.
    5. Only output the story.
    """.format(text)

    # 编码输入
    inputs = story_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(story_model.device)

    # 【关键修复】添加 no_repeat_ngram_size=2，彻底禁止重复
    outputs = story_model.generate(
        **inputs,
        max_new_tokens=100,  # 只生成新内容，不含Prompt
        min_new_tokens=50,
        temperature=0.7,
        top_p=0.85,
        no_repeat_ngram_size=2,  # 核心：禁止2个词以上的重复序列
        do_sample=True,
        num_beams=3,
        early_stopping=True
    )

    # 解码并清理
    story = story_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 最终兜底：如果还残留Prompt，直接截断（T5偶尔会这样）
    if "Generate a fun story" in story:
        story = story.split("output the story.")[-1].strip()

    # 字数控制
    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100]) + "!"
    elif len(words) < 50:
        story += " They all cheered and promised to play again tomorrow!"

    return story

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
