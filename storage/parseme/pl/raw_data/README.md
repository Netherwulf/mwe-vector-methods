README
------
This is the README file for the PARSEME verbal multiword expressions (VMWEs) corpus for Polish, edition 1.2. See the wiki pages of the [PARSEME corpora](https://gitlab.com/parseme/corpora/-/wikis/) initiative for the full documentation of the annotation principles.

The present Polish data result from an update and an extension of the Polish part of the [PARSEME 1.1 corpus](http://hdl.handle.net/11372/LRT-2842). 
Changes with respect to the 1.1 version are the following:
* updating the morphosyntactic annotation (UPOS and FEATS columns) to make them compatible with the [Universal Dependencies version 2.5](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3105)
* annotating new files, with sentences having identifiers starting with 110- (books), 200- (fiction),  320- (science and manuals)
* eliminating all annotations of the IAV category (which was annotated experimentally in edition 1.1.)
* fixing known bugs in previous annotations
* providing a companion raw corpus, automatically annotated for morpho-syntax

The raw corpus is not released in the present directory, but can be downloaded from a [dedicated page](https://gitlab.com/parseme/corpora/-/wikis/Raw-corpora-for-the-PARSEME-1.2-shared-task)

Source corpora
-------
All the annotated data come from one of these (more or less overlapping) sources:
* `NKJP`: 1-million word manually annotated subcorpus of the [National Corpus of Polish](http://clip.ipipan.waw.pl/NationalCorpusOfPolish) - most texts (i.e. those not in PDB-UD) whose identifiers start with 130 (daily newspapers) are included. 
* `PCC`: [Polish Coreference Corpus](http://zil.ipipan.waw.pl/PolishCoreferenceCorpus) (version 0.92) - the 21 "long" texts from this corpus are included, 36,000 tokens, Rzeczpospolita newspaper
* `PDB-UD`: Polish Dependency Bank, part of the [UD corpus version 2.5](http://hdl.handle.net/11234/1-3105); an up-graded version of the [PDB](http://zil.ipipan.waw.pl/PDB) corpus, containing notably [Składnica](http://zil.ipipan.waw.pl/Sk%C5%82adnica), converted into the dependency format -  sentences having the following identifiers are included:
  * starting with 110 (literature), 200 (fiction), 120 (periodicals), 130 (daily newspapers), 310 (non-fiction), 320 (popular science), 330 (manuals)
  * including names of newspapers (e.g. BrukowiecOchlanski, EkspressWieczorny, GazetaGoleniowska, GazetaKociewska, GazetaLubuska, GazetaMalborska, GazetaPomorska, GazetaTczewska, GazetaWroclawska, GlosPomorza, GlosSzczecinski, KurierKwidzynski, KurierSzczecinski, NIE, NowaTrybunaOpolska, Rzeczpospolita, SlowoPowszechne, SuperExpress, TrybunaLudu, TrybunaSlaska, ZycieINowoczesnosc, ZycieWarszawy)
  * including OTHER (mixture of texts from newspapers, social media and parliamentary debates)

Format
--------------------
The data are in the [.cupt](http://multiword.sourceforge.net/cupt-format) format. The following tagsets are used:
* column 4 (UPOS): [UD POS-tags](http://universaldependencies.org/u/pos) version 2.5 (as of March 2020), 
* column 5 (XPOS): [NKJP tagset](http://nkjp.pl/poliqarp/help/ense2.html) for sentences with source_sent_id containing *NationalCorpusOfPolish (and 130-)* or *PolishCoreferenceCorpus*; [PDB-UD](https://github.com/UniversalDependencies/UD_Polish-PDB) tagset for sentences containing *UD_Polish-PDB*. Both tagsets are roughly equivalent with a few exceptions.
* column 6 (FEATS): [UD features](http://universaldependencies.org/u/feat/index.html) version 2.5 (as of March 2020)
* column 8 (DEPREL): [UD dependency relations](http://universaldependencies.org/u/dep) version 2.5 (as of March 2020)
* column 11 (PARSEME:MWE): [PARSEME VMWE categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.2/?page=030_Categories_of_VMWEs) version 1.2

Text genres, origins and annotations
---------------------------
The text genre and source corpus can be recognized by the source sentence identifiers (source_sent_id and orig_file_sentence). 

The VMWE annotations (column 11) were performed by a single annotator. The following [categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.2/?page=030_Categories_of_VMWEs) are used: IAV, IRV, LVC.cause, LVC.full, VID. 

The morpho-syntactic annotation was partly automatic and partly manual, as shown in the following table:

| source_sent_id | orig_file_sentence | Genre | Source corpus | # sentences | Tokenization | LEMMA+XPOS | UPOS+FEATS | HEAD+DEPREL | PARSEME:MWE |
| -------------- | ------------------ | ----- | ------------- | ----------- | ------------ | ---------- | ---------- | ----------- | ----------- |
| contains *NationalCorpusOfPolish*  (and 130-)| | daily newspapers | NKJP | 9241 | converted from manual | manual | converted from manual | automatic<sup>*</sup> | manual |
| contains *PolishCoreferenceCorpus* | | daily newspapers | PCC | 2119 | automatic<sup>*</sup> | automatic<sup>*</sup> | automatic<sup>*</sup> | automatic<sup>*</sup> | manual |
| contains *UD_Polish-PDB* | starts with 110 | literature | PDB-UD | 201 |  converted from manual | manual | converted from manual | converted from manual | manual |
| contains *UD_Polish-PDB* | starts with 200 | fiction | PDB-UD | 2976 |  converted from manual | manual | converted from manual | converted from manual | manual |
| contains *UD_Polish-PDB* | starts with 120 | periodicals | PDB-UD | 1391 |  converted from manual | manual | converted from manual | converted from manual | manual |
| contains *UD_Polish-PDB* | starts with 130 | daily newspapers | PDB-UD | 2682 |  converted from manual | manual | converted from manual | converted from manual | manual |
| contains *UD_Polish-PDB* | starts with 310 | non-fiction | PDB-UD | 231 |  converted from manual | manual | converted from manual | converted from manual | manual |
| contains *UD_Polish-PDB* | starts with 320 | popular science | PDB-UD | 229 |  converted from manual | manual | converted from manual | converted from manual | manual |
| contains *UD_Polish-PDB* | starts with 330 | manuals | PDB-UD | 457 |  converted from manual | manual | converted from manual | converted from manual | manual |
| contains *UD_Polish-PDB* | contains a name from the newspaper list for PDB in [Corpora](#corpora) | newspapers | PDB-UD | 2167 |  converted from manual | manual | converted from manual | converted from manual | manual |
| contains *UD_Polish-PDB* | contains *OTHER* | newspapers, social media and parliamentary debates | PDB-UD | 1853 | converted from manual | manual | manual | manual | manual |

<sup>*</sup> automatic annotation signaled in the table above was performed with [UDPipe](http://ufal.mff.cuni.cz/udpipe) using the [polish-pdb-ud-2.5-191206](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/polish-pdb-ud-2.5-191206.udpipe?sequence=76&isAllowed=y) model.

Companion raw corpus
--------------------
The manually annotated corpus, described above, is accompanied by a large "raw" corpus (meant for automatic discovery of VMWEs), in which VMWEs are not annotated and morphosyntax is automatically tagged. Its characteristics are the following:
* size (uncompressed): 161 GB
* sentences: 159,115,022
* tokens: 1,902,279,431
* tokens/sentence: 11.96
* format: [CoNLL-U](https://universaldependencies.org/format.html)
* source: [CoNLL 2017 shared task raw corpus](http://hdl.handle.net/11234/1-1989) for Polish (see the [paper](https://www.aclweb.org/anthology/K17-3001.pdf#page=3)) 
* genre: Wikipedia pages and various other web pages (from CommonCrawl)
* morpho-syntactic tagging: upgraded to a [UD-2.5](http://hdl.handle.net/11234/1-3105)-compatible version with [UDPipe](http://ufal.mff.cuni.cz/udpipe) using the [polish-pdb-ud-2.5-191206](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/polish-pdb-ud-2.5-191206.udpipe?sequence=76&isAllowed=y) model (same as for the automatically tagged parts of the manually annotated corpus)
* compatibility of the raw corpus with the manually annotated corpus: same tagset and UdPipe model used.

Authors
----------
All VMWEs annotations (column 11) were performed by Agata Savary. For authorship of the data in columns 1-10 see the original corpora.
The conversion of the morphosyntactic annotations to the UD2.5-compatible format was performed by Jakub Waszczuk. 
The preparation of the companion raw corpus was also performed by Jakub Waszczuk.

License
----------
The VMWEs annotations (column 11) are distributed under the terms of the [CC-BY v4](https://creativecommons.org/licenses/by/4.0/) license.
The lemmas, POS-tags, morphological and features (columns 1-6), are distributed under the terms of the ([GNU GPL v.3](https://www.gnu.org/licenses/gpl.html)) for the sentences from NKJP, ([CC BY v.3](http://creativecommons.org/licenses/by/3.0/deed.en_US)) for the those from PCC, and [CC-BY-SA 0.4](https://creativecommons.org/licenses/by-sa/4.0/) for those from PDB-UD.
Dependency relations (columns 7-9) are distributed under the terms of the [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/) license.
The raw corpus is distributed under the terms of the [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/) license.

Contact
----------

*  agata.savary@univ-tours.fr
*  jakub.waszczuk@hhu.de
