from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from em_research_agentic.utils.state import AgentState
from langchain_core.pydantic_v1 import BaseModel
from langgraph.graph import END
from typing import List
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from em_research_agentic.utils.prompts import PLAN_PROMPT, RESEARCH_PLAN_PROMPT, WRITER_PROMPT, REFLECTION_PROMPT, RESEARCH_CRITIQUE_PROMPT
from em_research_agentic.utils.tools import tavily
from em_research_agentic.utils.article_summarizer import generate_summaries


class Queries(BaseModel):
    queries: List[str]


class SearchResponse(BaseModel):
    content: str
    url: str
    title: str

    def to_string(self):
        return f"Article title:{self.title}\n\nArticle Content:\n {self.content}\n\nURL: {self.url}"


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "anthropic":
        model = ChatAnthropic(
            temperature=0, model_name="claude-3-5-sonnet-20240620")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    return model


def plan_node(state: AgentState, config):

    input_message = f"The topic is:\n{state['task']}\n"

    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=input_message)
    ]
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)
    response = model.invoke(messages)
    return {"plan": response.content}

# Node for Research Planning


def research_plan_node(state: AgentState, config):
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)

    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])

    content = state['content'] or []
    max_results = config.get('configurable', {}).get('max_results_tavily', 1)
    for q in queries.queries:
        response = tavily.search(query=q, max_results=max_results)
        # TODO: Change this to better handle large articles
        article_urls = [r['url'] for r in response['results']][:1]
        summaries = generate_summaries(article_urls)
        for r, summary in zip(response['results'], summaries):
            response_obj = SearchResponse(
                content=summary, url=r['url'], title=r['title'])
            content.append(response_obj.to_string())
    return {"content": content}

# Node for Writing


def generation_node(state: AgentState, config):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
    ]
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)
    response = model.invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1
    }

# Node for reflection


def reflection_node(state: AgentState, config):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)
    response = model.invoke(messages)
    return {"critique": response.content}

# Node for Research Critique


def research_critique_node(state: AgentState, config):
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []

    max_results = config.get('configurable', {}).get('max_results_tavily', 1)

    for q in queries.queries:
        response = tavily.search(query=q, max_results=max_results)
        # TODO: Change this to better handle large articles
        article_urls = [r['url'] for r in response['results']][:1]
        summaries = generate_summaries(article_urls)
        for r, summary in zip(response['results'], summaries):
            response_obj = SearchResponse(
                content=summary, url=r['url'], title=r['title'])
            content.append(response_obj.to_string())
    return {"content": content}

# Edge


def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"
