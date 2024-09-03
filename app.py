import streamlit as st

import os

import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
from langchain import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.chains import ConversationalRetrievalChain, LLMChain, ConversationChain
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from datetime import date
import time



# from langchain.llms import OpenAI
# from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()


if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        # output_key="answer",
        return_messages=True
        \
    )


if 'patient' not in st.session_state: 
    st.session_state['patient'] = []

if 'doctor' not in st.session_state:
    st.session_state['doctor'] = []

if 'page' not in st.session_state:
    st.session_state.page = "query_page"

if 'symptoms' not in st.session_state:
    st.session_state.symptoms = 'symptoms'

if 'patient_information' not in st.session_state:
    st.session_state.patient_information = []

if 'appoint' not in st.session_state:
    st.session_state.appoint = ''


class HealthAssistant:
    def __init__(self):
        pass

    def get_prompt(self):
        qa_prompt = PromptTemplate(
            template="""You are a healthcare chatbot designed to gather information about a patient's condition. Ask one or two questions at a time to make the conversation feel natural and manageable. Be empathetic and patient in your responses. After gathering key details, summarize the information.

            patient_query:{query}

            Start with a general question and gradually ask more specific questions as needed. check {chat_history} to ask follow up questions

            """,
            
            input_variables=['query']
        )
         
        eval_prompt = PromptTemplate(
            template="""you are a health specialist. you are provided with consversation between patient and a doctor.your job is to evaluate there conservation and determine the following things:
            
            1. what is patient suffering from
            2. How risky it is should the patient go to hospital?
            3. what kind of diet should patient follow
            4. Any other additional advice
            
            patient: {patient}
            doctor: {doctor}
            
            Instead of patient say you""",

            input_variables=['patient','doctor']
        )
        return qa_prompt,eval_prompt
    

    
    def get_chain(self, qa_prompt,eval_prompt):
        med_llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
        # med_llm = ChatGroq(
        #     model="llama3-groq-70b-8192-tool-use-preview",
        #     temperature=0.5,
        # )
        # med_llm = ChatOpenAI(
        #     model='gpt-3.5-turbo',
        #     api_key="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"
        #     )

        qa_chain = LLMChain(
            prompt=qa_prompt,
            llm = med_llm,
            memory = st.session_state.memory
        )
        
        eval_chain = LLMChain(
            prompt= eval_prompt,
            llm = med_llm
        )


        return qa_chain,  eval_chain
    
    def get_text_from_cv(self,pdf_file):
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
        splitted_document = text_splitter.split_text(text)

        return text
        


class StreamlitApp(HealthAssistant):
    def __init__(self):
        pass

    def show_history(self):
        for i in range(len(st.session_state['patient'])):
            with st.chat_message('user'):
                st.write(st.session_state['patient'][i])

            with st.chat_message('ai'):
                st.write(st.session_state['doctor'][i])

    def UI(self):
        def display_text(text, delay=0.3):
            """Display text one line at a time with a delay."""
            for line in text.split('\n'):
                st.write(line)
                time.sleep(delay)  # Adjust the delay as needed
                
        def query_page():
            st.title("HealthMate")
            st.subheader("AI Health Assistant")
            history_container = st.container()
            query_container = st.container()

            qa_prompt, eval_prompt = self.get_prompt()
            qa_chain, eval_chain = self.get_chain(qa_prompt, eval_prompt)

            query = query_container.chat_input('Enter your query')
            thanks = st.button("Thank You")
            
            if query:
                result = qa_chain.invoke({'query': query})
                print(result)
                st.session_state.patient.append(query)
                st.session_state.doctor.append(result['text'])

                with history_container:
                    self.show_history()

            if thanks:
                # Switch to evaluation page
                st.session_state.page = "eval_page"
                st.experimental_rerun()

        def eval_page():
            st.title("Medical Report")
            
            # Check if results are already in session state
            if 'summarized_report' not in st.session_state:
                med_llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)
                report_prompt = PromptTemplate(
                    template="""You are provided with the medical report. Your job is to find the symptoms of the patient from that report and summarize it in 2 to 3 lines. Return the summary only; don't return any other thing.

                    medical_report: {report}""",
                    input_variables=['report']
                )
                qa_prompt, eval_prompt = self.get_prompt()
                qa_chain, eval_chain = self.get_chain(qa_prompt, eval_prompt)
                report_chain = LLMChain(prompt=report_prompt, llm=med_llm)

                result = eval_chain.apply([{
                    'patient': st.session_state.patient,
                    'doctor': st.session_state.doctor
                }])
                summarized_report = report_chain.invoke({'report': result[0]['text']})
                st.session_state.symptoms = summarized_report
                st.session_state.result_text = result[0]['text']  # Store result text in session state
                st.session_state.summarized_report = summarized_report  # Store summarized report in session state
            else:
                # Use the stored values
                result_text = st.session_state.result_text
                summarized_report = st.session_state.summarized_report
                st.session_state.symptoms = summarized_report

            display_text(st.session_state.result_text)  # Use stored result text

            # Add a button to return to appoint a doctor
            st.subheader("Do you need a Doctor?")
            col1, col2 = st.columns(2)  # Create two columns
            with col1:
                yes_button = st.button("Yes")  # Place the "Yes" button in the first column

            with col2:
                no_button = st.button("No")

            if yes_button:
                st.session_state.page = "hospitals_list"
                st.experimental_rerun()

            if no_button:
                st.session_state.page = "thank_you"
                st.rerun()

        def appointment_form():
            st.title("Appointment Form")

            # Create the form
            with st.form("booking_form"):
                name = st.text_input("Name")
                address = st.text_area("Address")
                ticket_price = st.number_input("Ticket Price", min_value=0.0, format="%.2f")
                appointment_date = st.date_input("Select Appointment Date", value=date.today())
                
                # Submit button
                submit_button = st.form_submit_button(label="Book Token")

            # Handle form submission
            if submit_button:
                st.success(f"Token booked successfully for {name} on {appointment_date}.")
                st.write(f"Address: {address}")


                st.write(f"Ticket Price: ${ticket_price:.2f}")

                st.session_state.page = "end"
                st.rerun()
        
        def get_text_from_pdf(pdf_file):
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 150, chunk_overlap = 20)
            splitted_document = text_splitter.split_text(text)

            return splitted_document

        def hospitals_list():
            st.title("Hospital List")
            hospital_info = get_text_from_pdf('Hospital_Information.pdf')

            # llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0) 
            llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
            
            doc_prompt = PromptTemplate(
                template="""You are provided with the symptoms of a patient along with a list of hospitals. 
                Your job is to find one hospital if available where the patient can receive treatment.

                Patient Symptoms: {symptoms}
                Hospitals List: {hospitals_list}

                return the information of at least two hospital in structured format.
                example:
                hospital name: write hospital name here
                description: write description of hospital here
                doctor_availiblity:24/7
                location: location of hospital
                doctor_list: enlist doctors names

                you need to provide this much information only
                """,
                input_variables=['symptoms', 'hospitals_list']
            )

            qa_chain = LLMChain(
                prompt=doc_prompt,  # Corrected to 'prompt'
                llm=llm
            )

            for i in range(2):
                result = qa_chain.apply([{
                    'symptoms': st.session_state.symptoms,
                    'hospitals_list': hospital_info  # Corrected 'hospitals_info' to 'hospitals_list'
                }])
                
                # Create a container to act as a card
                with st.container():
                    # Display card content
                    st.markdown(
                        """
                        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; margin-bottom: 20px; background-color: #f9f9f9; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                            <p style="font-size: 16px; color: #333;">{}</p>
                        </div>
                        """.format(result[0]['text']),
                        unsafe_allow_html=True
                    )
                    
                    # Place the button inside the card
                    if st.button("Appoint", key=f"appoint_{i}"):
                        st.session_state.page = "appointment_form"
                        st.rerun()


        def thank_you():
            st.title("Thank you for visiting our website")

        def booked():
            st.title("Your appointment has been booked")

        # Page Navigation
        if st.session_state.page == "query_page":
            query_page()
        elif st.session_state.page == "eval_page":
            eval_page()
        elif st.session_state.page == "hospitals_list":
            hospitals_list()
        elif st.session_state.page == "appointment_form":
            appointment_form()
        elif st.session_state.page == "end":
            booked()
        elif st.session_state.page == "thank_you":
            thank_you()



if __name__ == "__main__":
    app = StreamlitApp()
    app.UI()