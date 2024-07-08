import os
import openai
from pydantic import BaseModel, Field
from typing import List
from llama_index.readers.file import UnstructuredReader
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
import logging
import sys
import json
import nltk


nltk.download('averaged_perceptron_tagger')


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


openai.api_key = os.environ["OPENAI_API_KEY"]


class Instrument(BaseModel):
    """Datamodel for ticker trading details."""

    direction: str = Field(description="ticker trading - Buy, Sell, Hold etc")
    ticker: str = Field(
        description="Stock Ticker. 1-4 character code. Example: AAPL, TSLS, MSFT, VZ"
    )
    company_name: str = Field(
        description="Company name corresponding to ticker"
    )
    shares_traded: float = Field(description="Number of shares traded")
    percent_of_etf: float = Field(description="Percentage of ETF")


class Etf(BaseModel):
    """ETF trading data model"""

    etf_ticker: str = Field(
        description="ETF Ticker code. Example: ARKK, FSPTX"
    )
    trade_date: str = Field(description="Date of trading")
    stocks: List[Instrument] = Field(
        description="List of instruments or shares traded under this etf"
    )


class EmailData(BaseModel):
    """Data model for email extracted information."""

    etfs: List[Etf] = Field(
        description="List of ETFs described in email having list of shares traded under it"
    )
    trade_notification_date: str = Field(
        description="Date of trade notification"
    )
    sender_email_id: str = Field(description="Email Id of the email sender.")
    email_date_time: str = Field(description="Date and time of email")


# Initialize the UnstructuredReader
loader = UnstructuredReader()

# For eml file
eml_documents = loader.load_data("tasks.eml")
email_content = eml_documents[0].text
print("\n\n Email contents")
print(email_content)


prompt = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role="system",
            content=(
                "You are an expert assitant for extracting insights from email in JSON format. \n"
                "You extract data and returns it in JSON format, according to provided JSON schema, from given email message. \n"
                "REMEMBER to return extracted data only from provided email message."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Email Message: \n" "------\n" "{email_msg_content}\n" "------"
            ),
        ),
    ]
)

llm = OpenAI(model="gpt-4o")

program = OpenAIPydanticProgram.from_defaults(
    output_cls=EmailData,
    llm=llm,
    prompt=prompt,
    verbose=True,
)


output = program(email_msg_content=email_content)
print("Output JSON From .eml File: ")
print(json.dumps(output.dict(), indent=2))