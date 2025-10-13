##install validators==0.28.1
##install youtube_transcript_api
import validators,streamlit  as st 
from langchain import PromptTemplate
from langchain_groq import ChatGroq 
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv
import os

load_dotenv()

hf_key=os.getenv("GROQ_API_KEY")
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=150,temperature=0.7,token=os.getenv("HF_TOKEN"))

#llm=ChatGroq(groq_api_key=hf_key,model="llama-3.1-8b-instant")

prompt_template="""provide the summary of the following content in 300 words:
Content:{text}"""

prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

##streamlit app
st.set_page_config(page_title="Langchain:Summarize text from Youtube or Website",page_icon="🦜")
st.title("🦜 Langchain:Summarize Text from Youtube or Website")
st.subheader("summarize URL")



##get the groq api key and field to be summarized
with st.sidebar:
    hf_key=st.text_input("Hugging_face_key",value="",type="password")

generic_url=st.text_input("url",label_visibility="collapsed")   ## collapse means it will not be highlighted
##create button

if st.button("summarize the content from Youtube and website"):
    ##validate all the inputs   ##strip means remove empty characters
    if not hf_key.strip() or not generic_url.strip():
        st.error("please provide the information")
    elif not  validators.url(generic_url):
        st.error("Please enter a valid URL. it can may be a yt video url or website url")                                                ##if url given then validate it
    else:
        try:
            with st.spinner("Waiting ..."):
                ##loading the website data or yt video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=False)  ##add video info is true means --  youtube_transcript_api only works if the video has subtitles or captions available.  so set it false in case of issues
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,headers={"user-agent":"mozilla..."})
                docs=loader.load()
                ##chain for summarization

                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)
                st.success(output_summary)
        except Exception as e:
            #st.error(f"Exception: {e}")
            st.exception(f"Exception:{e}")      



##install pytube for summarizing you tube video






