import uuid
from abc import ABC, abstractmethod
from typing import NamedTuple


class ReRanker[T](ABC):
    @abstractmethod
    def rerank(self) -> list[T]:
        raise NotImplementedError


class ReciprocalRankFusionDocument[T](NamedTuple):
    id: uuid.UUID | int | str
    document: T


class ReciprocalRankFusionSource[T](NamedTuple):
    k: int
    ranking_document_list: list[ReciprocalRankFusionDocument[T]]


class ScoredRankingDocument[T](NamedTuple):
    ranking_document: ReciprocalRankFusionDocument[T]
    score: float = 0.0


class ReciprocalRankFusionReRanker[T](
    ReRanker[ReciprocalRankFusionDocument[T]]
):
    def __init__(self, sources: list[ReciprocalRankFusionSource[T]]) -> None:
        self.sources = sources

    def rerank(self) -> list[ReciprocalRankFusionDocument[T]]:
        document_id_dict: dict[
            uuid.UUID | int | str, ScoredRankingDocument[T]
        ] = {}
        for source in self.sources:
            for ranking_document in source.ranking_document_list:
                document_id_dict[ranking_document.id] = ScoredRankingDocument(
                    ranking_document=ranking_document,
                )

        for embedding_id in document_id_dict:
            score = 0.0
            for source in self.sources:
                for index, ranking_document in enumerate(
                    source.ranking_document_list
                ):
                    if embedding_id == ranking_document.id:
                        score += 1.0 / (source.k + index + 1)
            document_id_dict[embedding_id] = ScoredRankingDocument(
                score=score,
                ranking_document=document_id_dict[
                    embedding_id
                ].ranking_document,
            )

        sorted_embeddings_id_dict = sorted(
            filter(lambda item: item[1].score > 0, document_id_dict.items()),
            key=lambda item: item[1].score,
            reverse=True,
        )

        return [item[1].ranking_document for item in sorted_embeddings_id_dict]
