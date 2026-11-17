# SELEX

We wanted to create a set of zebrafish transcription factors (TFs) that had completely unknown motifs in their human or mouse orthologs in JASPAR 2022. In AlphaFold3's training on transcription factors bound to DNA, they used TF profiles from Jaspar 9 (2022) with genes who had experimental DNA sequence binding specificity evidence from two high-throughput SELEX datasets (Abrahmson et al., 2024, Supplementary Information). If a gene did not meet this requirement, we take it as this TF was not used in their TF positive distillation set.

We asked ChatGPT Deep Research to give us a list of TFs with completely unknown motifs in any organism, as well as asking Gemini to refine this list. In the end we chose around 30 TFs to proceed with, and they can be split into three groups:
1. Zebrafish TFs with evidence for a motif in human or mouse ortholog in JASPAR 2022. 
2. Zebrafish TFs with motif evidence in human or mouse ortholog in JASPAR 2022, but DNA-binding domain homology is not 100%. 
3. Zebrafish TFs with motif evidence in human or mouse ortholog in JASPAR 2022, and homology of the DNA-binding domain is 100%.

We also checked if these TFs had relatively high confidence scores, both in the DNA binding domain and in the interaction between the protein and DNA. All of the TFs returned from our search can be found in `selex_candidates` .csv files, and the accepted candidates can be found in the `selex` .csv files. The sequences for these proteins were retreived by running `data/get_seqs.py`.