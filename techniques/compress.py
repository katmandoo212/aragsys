from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
import re

if TYPE_CHECKING:
    from ollama.client import OllamaClient

@dataclass
class CompressTechnique:
    ollama_client: "OllamaClient"
    use_llm_refinement: bool
    min_keyword_matches: int
    segment_length: int
    top_k_segments: int
    base_technique: Optional[object] = None

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve and compress documents to relevant segments."""
        # Get initial results from base technique
        if self.base_technique:
            results = self.base_technique.retrieve(query)
        else:
            return []

        if not results:
            return []

        compressed_results = []
        for result in results:
            compressed = self._compress_document(query, result["content"])
            if compressed:
                result["content"] = compressed
                compressed_results.append(result)

        return compressed_results

    def _compress_document(self, query: str, content: str) -> str:
        """Compress document using keyword extraction and optional LLM refinement."""
        # Extract query terms (simple tokenization, lowercase)
        query_terms = set(re.findall(r'\w+', query.lower()))

        # Split into segments (simple sentence-based)
        segments = re.split(r'[.!?]+', content)
        segments = [s.strip() for s in segments if s.strip()]

        # Score segments by keyword overlap
        scored_segments = []
        for segment in segments:
            segment_terms = set(re.findall(r'\w+', segment.lower()))
            overlap = len(query_terms & segment_terms)

            if overlap >= self.min_keyword_matches:
                scored_segments.append((segment, overlap))

        # Sort by overlap and keep top_k
        scored_segments.sort(key=lambda x: x[1], reverse=True)
        top_segments = [s[0] for s in scored_segments[:self.top_k_segments]]

        if not top_segments:
            return content

        # Optional LLM refinement
        if self.use_llm_refinement:
            return self._llm_refine(query, top_segments)

        # Join segments without LLM
        return ". ".join(top_segments)

    def _llm_refine(self, query: str, segments: list[str]) -> str:
        """Refine segments using LLM to keep only relevant content."""
        segments_text = "\n".join(f"- {s}" for s in segments)
        prompt = (
            f"From these segments, extract only content directly relevant to the query.\n"
            f"Query: {query}\n"
            f"Segments:\n{segments_text}\n"
            f"Relevant content:"
        )
        return self.ollama_client.generate(prompt, "glm-4.7:cloud")