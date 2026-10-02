# FreshStack Laravel findings (24 native-Markdown docs, 73 questions; qwen2.5:7b; computed 2026-10-02 20:52)

Primary metric prec@1024t (share of a 1,024-token budget that is gold text); primary retrievers
hybrid and BM25; pooled-corpus retrieval; bootstrap over questions. Full grid: results/freshstack.json.

- RQ1 structure: struct/text vs fixedtok: no difference on any retriever (hybrid -0.026 [-0.064,+0.012]);
  vs fixed512: better under hybrid (+0.051 [+0.014,+0.090]) and nomic (+0.065); BM25/mxbai CIs touch 0.
- RQ2 enrichment vs title-chain: hybrid +0.023 [-0.016,+0.061], BM25 +0.010 [-0.019,+0.038]: no difference.
  Secondary retriever mxbai: +0.046 [+0.006,+0.086].
- RQ3 enrichment vs contextual retrieval: hybrid +0.017 ns, BM25 +0.005 ns; mxbai +0.067 [+0.030,+0.105].
- RQ4b merging: merged_rk/meta - enr_rk/meta ~0 on all retrievers. (No rolling-key ablation on FreshStack.)
- RQ4a mechanism (with rolling keys only): key reuse 8.6%, 116 chunks share a key, 56 removed by merging.
- Exploratory: gains for enrichment appear only with mxbai, whose 512-token input limit truncates chunk
  text; the generated title/summary placed first survives truncation (hypothesis).
- Cost per chunk: enrichment 1,065 in / 168 out tokens; contextual retrieval 9,825 in / 48 out
  (Laravel docs are long; CR reads the whole document per call).
- Validation: 0 warnings.
