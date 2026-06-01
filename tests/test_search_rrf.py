import unittest

from src.search import BM25Hit, Hit, merge_hits_by_rrf


class MergeHitsByRRFTest(unittest.TestCase):
    def test_chunk_found_by_both_searches_ranks_first(self) -> None:
        # Chunk B есть и в vector, и в BM25 выдаче, поэтому его RRF-баллы складываются.
        vector_hits = [
            Hit(source="A.txt", chunk_index=0, document="vector only", distance=0.2),
            Hit(source="B.txt", chunk_index=0, document="both", distance=0.3),
        ]
        bm25_hits = [
            BM25Hit(source="B.txt", chunk_index=0, document="both", score=10.0),
            BM25Hit(source="C.txt", chunk_index=0, document="bm25 only", score=5.0),
        ]

        hybrid_hits = merge_hits_by_rrf(vector_hits, bm25_hits)

        self.assertEqual(hybrid_hits[0].source, "B.txt")
        self.assertEqual(hybrid_hits[0].chunk_index, 0)
        self.assertEqual(hybrid_hits[0].distance, 0.3)
        self.assertEqual(hybrid_hits[0].bm25_score, 10.0)

    def test_equal_single_rank_prefers_vector_hit_by_tie_breaker(self) -> None:
        # Если один chunk первый только в vector, а другой первый только в BM25,
        # RRF-score одинаковый. Тогда текущий tie-breaker предпочитает vector distance.
        vector_hits = [
            Hit(source="A.txt", chunk_index=0, document="vector only", distance=0.2),
        ]
        bm25_hits = [
            BM25Hit(source="B.txt", chunk_index=0, document="bm25 only", score=10.0),
        ]

        hybrid_hits = merge_hits_by_rrf(vector_hits, bm25_hits)

        self.assertAlmostEqual(hybrid_hits[0].rrf_score, hybrid_hits[1].rrf_score)
        self.assertEqual(hybrid_hits[0].source, "A.txt")
        self.assertEqual(hybrid_hits[0].bm25_score, None)
        self.assertEqual(hybrid_hits[1].source, "B.txt")
        self.assertEqual(hybrid_hits[1].distance, None)


if __name__ == "__main__":
    unittest.main()
