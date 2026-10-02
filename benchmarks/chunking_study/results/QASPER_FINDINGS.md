# Qasper findings (30 papers, 79 questions; qwen2.5:7b; computed 2026-10-02 18:13)

Primary metric rec@512t; primary retrievers hybrid and BM25 (ANALYSIS_PLAN.md).
Full grids: results/qasper.json, results/qasper.txt; mechanism/cost: results/qasper_analysis.json.

- RQ1 structure: struct/text > fixed512 on all retrievers (+0.10..+0.23, all CIs exclude 0);
  > fixedtok under hybrid (+0.127 [+0.053,+0.203]) and nomic (+0.104), not BM25/mxbai.
- RQ2 enrichment vs title-chain: hybrid -0.023 [-0.084,+0.045]; BM25 +0.068 [-0.006,+0.150]. No difference.
- RQ3 enrichment vs contextual retrieval: no difference on any retriever (hybrid -0.041 [-0.101,+0.023]).
- RQ4a mechanism (paired, 30 papers): key reuse 14.7% (rk) vs 5.5% (nork); chunks sharing a key 185 vs 72;
  removed by merging 94 vs 36; 90% of chunks list >=1 related key.
- RQ4b retrieval: merged_rk/meta - merged_nork/meta: BM25 -0.060 [-0.131,-0.002] (worse), hybrid -0.031 ns;
  merged_rk/meta - enr_rk/meta: no difference.
- Exploratory (not pre-specified): under BM25 alone, every LLM-prefixed index beats struct/text
  (enr_nork/meta 0.416 vs 0.311); under hybrid none does. struct/tc ~ struct/text (generic Qasper headers).
- Cost per chunk: enrichment 1,002 in / 188 out tokens; contextual retrieval 5,520 in / 58 out.
- 1 of 715 enr_rk chunks lacked key/title/summary (model miss, kept).
