import os
from copy import deepcopy
from typing import Any, Dict, List, Union
from loguru import logger
import streamlit as st
import uuid
import json

from langchain import LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentType
from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from pydantic import BaseModel, Field
import re
from langchain.output_parsers import PydanticOutputParser

from Real_estate.chains import SalesConversationChain, StageAnalyzerChain
from Real_estate.logger import time_logger
from Real_estate.parsers import SalesConvoOutputParser
from Real_estate.prompts import SALES_AGENT_TOOLS_PROMPT
from Real_estate.stages import CONVERSATION_STAGES
from Real_estate.templates import CustomPromptTemplateForTools
from Real_estate.tools import get_tools, setup_knowledge_base
from langchain.callbacks import get_openai_callback

from azure.cosmos import CosmosClient
from streamlit.runtime.scriptrunner import add_script_run_ctx

cosmosdb_endpoint = "https://brad-cosmos.documents.azure.com:443/"
if os.getenv("cosmoskey"):
    cosmosdb_key = os.getenv("cosmoskey")
else:
    cosmosdb_key = st.secrets["cosmosdb_key"]
cosmosdb_database_name = "RealEstate"
cosmosdb_container_name = "UserChatHistory"

client = CosmosClient(cosmosdb_endpoint, cosmosdb_key)

# Get database reference
database = client.get_database_client(cosmosdb_database_name)

# Get container reference
container = database.get_container_client(cosmosdb_container_name)

import streamlit as st

from Real_estate.callbackhandler import MyCustomHandler


def add_newlines_around_tag(text, tag="<tag>"):
    pattern = rf'({re.escape(tag)})'
    replaced_text = re.sub(pattern, r'\n\n\n\1\n', text)
    return replaced_text


class Real_estate(Chain, BaseModel):
    """Controller model for the Sales Agent."""

    conversation_history: List[str] = []
    conversation_stage_id: str = "1"
    current_conversation_stage: str = CONVERSATION_STAGES.get("1")
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    sales_agent_executor: Union[AgentExecutor, None] = Field(...)
    knowledge_base: Union[RetrievalQA, None] = Field(...)
    sales_conversation_utterance_chain: SalesConversationChain = Field(...)
    conversation_stage_dict: Dict = CONVERSATION_STAGES

    use_tools: bool = False
    salesperson_name: str = "Ted Lasso"
    salesperson_role: str = "Business Development Representative"
    company_name: str = "Sleep Haven"
    company_business: str = "Sleep Haven is a premium mattress company that provides customers with the most comfortable and supportive sleeping experience possible. We offer a range of high-quality mattresses, pillows, and bedding accessories that are designed to meet the unique needs of our customers."
    company_values: str = "Our mission at Sleep Haven is to help people achieve a better night's sleep by providing them with the best possible sleep solutions. We believe that quality sleep is essential to overall health and well-being, and we are committed to helping our customers achieve optimal sleep by offering exceptional products and customer service."
    conversation_purpose: str = "find out whether they are looking to achieve better sleep via buying a premier mattress."
    conversation_type: str = "call"

    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, "1")

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    @time_logger
    def seed_agent(self):
        # Step 1: seed the conversation
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []

    @time_logger
    def determine_conversation_stage(self):
        self.conversation_stage_id = self.stage_analyzer_chain.run(
            conversation_history="\n".join(self.conversation_history).rstrip("\n"),
            conversation_stage_id=self.conversation_stage_id,
            conversation_stages="\n".join(
                [
                    str(key) + ": " + str(value)
                    for key, value in CONVERSATION_STAGES.items()
                ]
            ),
        )

        print(f"Conversation Stage ID: {self.conversation_stage_id}")
        self.current_conversation_stage = self.retrieve_conversation_stage(
            self.conversation_stage_id
        )

        print(f"Conversation Stage: {self.current_conversation_stage}")

    def human_step(self, human_input):
        # process human input
        human_input = "User: " + human_input + " <END_OF_TURN>"
        self.conversation_history.append(human_input)

    @time_logger
    def step(
            self, return_streaming_generator: bool = False, model_name="gpt-35-turbo", summary: bool = False):
        """
        Args:
            return_streaming_generator (bool): whether or not return
            streaming generator object to manipulate streaming chunks in downstream applications.
        """
        if not return_streaming_generator:
            self._call(inputs={}, summary=summary)
        else:
            return self._streaming_generator(model_name=model_name)

    # TO-DO change this override "run" override the "run method" in the SalesConversation chain!
    @time_logger
    def _streaming_generator(self, model_name="gpt-3.5-turbo-0613"):
        """
        Sometimes, the sales agent wants to take an action before the full LLM output is available.
        For instance, if we want to do text to speech on the partial LLM output.

        This function returns a streaming generator which can manipulate partial output from an LLM
        in-flight of the generation.

        Example:

        >> streaming_generator = self._streaming_generator()
        # Now I can loop through the output in chunks:
        >> for chunk in streaming_generator:
        Out: Chunk 1, Chunk 2, ... etc.
        See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
        """
        prompt = self.sales_conversation_utterance_chain.prep_prompts(
            [
                dict(
                    conversation_stage=self.current_conversation_stage,
                    conversation_history="\n".join(self.conversation_history),
                    salesperson_name=self.salesperson_name,
                    salesperson_role=self.salesperson_role,
                    company_name=self.company_name,
                    company_business=self.company_business,
                    company_values=self.company_values,
                    conversation_purpose=self.conversation_purpose,
                    conversation_type=self.conversation_type,
                )
            ]
        )

        inception_messages = prompt[0][0].to_messages()

        message_dict = {"role": "system", "content": inception_messages[0].content}

        if self.sales_conversation_utterance_chain.verbose:
            print("\033[92m" + inception_messages[0].content + "\033[0m")
        messages = [message_dict]

        return self.sales_conversation_utterance_chain.llm.completion_with_retry(
            messages=messages,
            stop="<END_OF_TURN>",
            stream=True,
            engine="text-embedding-ada-002"
        )

    def _call(self, inputs: Dict[str, Any], summary) -> None:
        """Run one step of the sales agent."""

        # Generate agent's utterance
        # if use tools
        try:
            if self.use_tools:
                logger.info('Used Tools Agent Executor')
                with get_openai_callback() as cb:
                    ai_message = self.sales_agent_executor.run(
                        input="",
                        conversation_stage=self.current_conversation_stage,
                        conversation_history="\n".join(self.conversation_history),
                        salesperson_name=self.salesperson_name,
                        salesperson_role=self.salesperson_role,
                        company_name=self.company_name,
                        company_business=self.company_business,
                        company_values=self.company_values,
                        conversation_purpose=self.conversation_purpose,
                        conversation_type=self.conversation_type,
                    )

                    document_id = str(uuid.uuid4())
                    property_data = {"id": document_id, "sessionId": document_id, "prompt": self.conversation_history,
                                     "response": ai_message, "total_tokens": cb.total_tokens}
                    #   # Insert document into Cosmos DB
                    container.upsert_item(body=property_data)

            else:
                # else
                logger.info('Used Tools Chain Executor')
                with get_openai_callback() as cb:
                    ai_message = self.sales_conversation_utterance_chain.run(
                        conversation_stage=self.current_conversation_stage,
                        conversation_history="\n".join(self.conversation_history),
                        salesperson_name=self.salesperson_name,
                        salesperson_role=self.salesperson_role,
                        company_name=self.company_name,
                        company_business=self.company_business,
                        company_values=self.company_values,
                        conversation_purpose=self.conversation_purpose,
                        conversation_type=self.conversation_type
                    )
                    document_id = str(uuid.uuid4())
                    property_data = {"id": document_id, "sessionId": document_id, "prompt": self.conversation_history,
                                     "response": ai_message, "total_tokens": cb.total_tokens}
                    # Insert document into Cosmos DB
                    container.upsert_item(body=property_data)

        except Exception as e:
            ai_message = ""
            logger.error('Chain Execute Error: ' + str(e))

        # Add agent's response to conversation history
        # print(f'{self.salesperson_name}: ', ai_message.rstrip('<END_OF_TURN>'))
        agent_name = self.salesperson_name
        ai_message = agent_name + ": " + ai_message
        as_msg = ai_message
        # print("As message"+as_msg)
        if '<END_OF_TURN>' in as_msg:
            as_msg = as_msg.replace('<END_OF_TURN>', '')
        st.session_state.chat_history.append(as_msg)
        if '<END_OF_TURN>' not in ai_message:
            ai_message += ' <END_OF_TURN>'
        self.conversation_history.append(ai_message)

        if summary is False:
            # print(ai_message.replace("<END_OF_TURN>", ""))
            for i, msg in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    pattern = r'https?://[^\s]+'
                    src_links = re.findall(pattern, msg)
                    links = []
                    for link in enumerate(src_links):
                        links.append(link)

                    msg = add_newlines_around_tag(msg)

                    #msg = re.sub(pattern, "", msg)
                    if links:
                        st.chat_message('user').markdown(msg, unsafe_allow_html=True)
                    else:
                        st.chat_message('user').write(msg)
                else:
                    st.chat_message('assistant').write(msg)

        return {}

    @classmethod
    @time_logger
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "Real_estate":
        """Initialize the Real_estate Controller."""
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)
        if (
                "use_custom_prompt" in kwargs.keys()
                and kwargs["use_custom_prompt"] == "True"
        ):
            print("Custom Prompt True")
            use_custom_prompt = deepcopy(kwargs["use_custom_prompt"])
            custom_prompt = deepcopy(kwargs["custom_prompt"])

            # clean up
            del kwargs["use_custom_prompt"]
            del kwargs["custom_prompt"]

            sales_conversation_utterance_chain = SalesConversationChain.from_llm(
                llm,
                verbose=verbose,
                use_custom_prompt=use_custom_prompt,
                custom_prompt=custom_prompt,
            )

        else:
            print("Custom Prompt False")

            sales_conversation_utterance_chain = SalesConversationChain.from_llm(
                llm, verbose=verbose
            )

        if "use_tools" in kwargs.keys() and kwargs["use_tools"] is True:
            # set up agent with tools
            print("Use Tools True")
            product_catalog = kwargs["product_catalog"]
            knowledge_base = setup_knowledge_base(product_catalog)
            tools = get_tools(knowledge_base)
            customhandler = MyCustomHandler()
            prompt = CustomPromptTemplateForTools(
                template=SALES_AGENT_TOOLS_PROMPT,
                tools_getter=lambda x: tools,
                # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
                # This includes the `intermediate_steps` variable because that is needed
                input_variables=[
                    "input",
                    "intermediate_steps",
                    "salesperson_name",
                    "salesperson_role",
                    "company_name",
                    "company_business",
                    "company_values",
                    "conversation_purpose",
                    "conversation_type",
                    "conversation_history",
                ],
                callbacks=[customhandler]
            )

            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose, callbacks=[customhandler])

            tool_names = [tool.name for tool in tools]

            # WARNING: this output parser is NOT reliable yet
            ## It makes assumptions about output from LLM which can break and throw an error
            output_parser = SalesConvoOutputParser(ai_prefix=kwargs["salesperson_name"])

            sales_agent_with_tools = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=verbose,
                handle_parsing_errors="Check your output and make sure it conforms!",
                max_execution_time=10,
                callbacks=[customhandler]
            )

            sales_agent_executor = AgentExecutor.from_agent_and_tools(
                agent=sales_agent_with_tools, tools=tools, verbose=verbose, callbacks=[customhandler]
            )
        else:
            sales_agent_executor = None
            knowledge_base = None

        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            sales_conversation_utterance_chain=sales_conversation_utterance_chain,
            sales_agent_executor=sales_agent_executor,
            knowledge_base=knowledge_base,
            verbose=verbose,
            **kwargs,
        )
