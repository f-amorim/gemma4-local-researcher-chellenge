from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from tavily import TavilyClient
from pydantic import BaseModel, Field
import pprint


class Descoberta (BaseModel):
    fato: str = Field(description="Resumo de um paper ou descoberta recente encontrada na web.")
    autores_ou_instituicao: str = Field(description="quem publicou ou instituião por tras da descoberta(se possível)")


class RelatorioPapers (BaseModel):
    """Estrutura da resposta a ser pesquisada com os campos:
     tema, principais_achados(esse sendo uma lista da classe Descobreta,
     e conclusão do agente)"""
    tema: str = Field(description="foco central da pesquisa solicitada")
    principais_achados: list[Descoberta] = Field("lista de artigos/papers maeados na busca")
    conclusao_agente: str = Field(description="uma sintese sobre o estado atual dessa pesquisa")

model = ChatOllama(model="gemma4:e2b", temperature= 0)


@tool
def pesquisa(question: str)->str:

    """esta função realiza uma pesquisa na web usando tavily com uma query passada para ela e retorna o maximo de 4 resultados"""
    tavily_client = TavilyClient(api_key = "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")

    resposta_busca = tavily_client.search(
        query = question,
        max_results= 4,
        include_images= False
    )

    return str(resposta_busca)


prompt = "Você é um agente de inteligẽncia sobre papers envolvendo IA.Você utiliza ferramentas de pesquisa sempre que necessario para manter as informções atalizadas e confiaveis"


agent =  create_agent(
    model = model,
    tools = [pesquisa],
    system_prompt = prompt,
    response_format = RelatorioPapers
    )


consulta ="ultimas pesquisas sobre o uso de outros modelos machine Learn além dos Transformers para a construção de llms"


resposta= agent.invoke({"messages": [{"role":"user", "content": consulta}]})




print(resposta)