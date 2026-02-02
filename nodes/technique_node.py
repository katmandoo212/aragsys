from dataclasses import dataclass
from pocketflow import Node


@dataclass
class TechniqueNode(Node):
    technique_name: str
    config: dict

    def prep(self, shared: dict):
        query = shared.get("query")
        docs = shared.get("retrieved_docs", [])
        return query, docs

    def exec(self, prep_result):
        return prep_result

    def post(self, shared: dict, prep_res, exec_res):
        return None