import streamlit as st
from langchain_community.llms import CTransformers

# Set Streamlit page title
st.set_page_config(page_title="📝 Blog Generator", layout="centered")

# App heading
st.title("Blog Generator using LLaMA 2")

# Input fields
topic = st.text_input("Enter Blog Topic")
audience = st.selectbox("Blog Target Audience", ["Students", "Researchers", "Developers", "General Readers"])
words = st.text_input("Number of words", placeholder="e.g., 100")

# Load LLaMA 2 model
@st.cache_resource
def load_model():
    return CTransformers(
        model=r"E:\Blog_Generation\Model\llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        config={"max_new_tokens": 512, "temperature": 0.7}
    )

llm = load_model()

# Generate blog content
if st.button("Generate Blog"):
    if topic and audience:
        with st.spinner("Generating blog..."):
            prompt = (
                f"Write a blog post on the topic '{topic}' for the audience '{audience}'. "
                f"The blog should be approximately {words} words long and should be engaging and informative."
            )
            result = llm.invoke(prompt)
            st.success("✅ Blog Generated!")
            st.text_area("✍️ Generated Blog", value=result, height=400)
    else:
        st.warning("Please enter both a topic and an audience.")
